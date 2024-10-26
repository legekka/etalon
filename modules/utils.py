import os
import json
import torch
import numpy as np
from accelerate import Accelerator
from datasets import load_dataset as load_dataset_hf

from modules.config import Config

import argparse

def get_class_weights(dataset, num_classes=None, label_column='label'):
    """
    Computes the class weights for a given dataset.

    Args:
        dataset (Dataset): A Hugging Face Dataset object.
        num_classes (int): The number of classes in the dataset.
        label_column (str): The column name containing the labels.

    Returns:
        torch.Tensor: 1D tensor containing the class weights.
    """
    labels = dataset[label_column]
    
    if num_classes is None:
        num_classes = len(set(labels))

    class_counts = np.zeros(num_classes)

    class_counts[class_counts == 0] = 1

    class_weights = np.sum(class_counts) / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.clamp(class_weights, min=0.05, max=5)

    return class_weights


def load_dataset(data_dir: str, split: str):
    """
    Loads a dataset from a local directory containing Parquet files or from the Hugging Face Hub.

    This function extends the Hugging Face `load_dataset` function by allowing
    loading datasets from local Parquet files. If the specified `data_dir` exists
    locally, it will load the dataset from the local Parquet files matching the specified `split`.
    Otherwise, it will attempt to load the dataset from the Hugging Face Hub using the given `data_dir`
    as the dataset identifier.

    Args:
        data_dir (str): The path to the dataset directory or the dataset identifier
                        on the Hugging Face Hub.
        split (str): The dataset split to load (e.g., 'train', 'test', 'validation').

    Returns:
        Dataset: A Hugging Face Dataset object.

    Raises:
        FileNotFoundError: If the local data directory does not exist and the dataset
                           cannot be loaded from the Hugging Face Hub.
        ValueError: If no Parquet files matching the split are found in the local directory.
    """
    # Check if data_dir exists locally
    if not os.path.exists(data_dir):
        # Attempt to load dataset from Hugging Face Hub
        return load_dataset_hf(data_dir, split=split)
    else:
        # List Parquet files matching the specified split
        parquet_files = os.listdir(data_dir)
        parquet_files = [f for f in parquet_files if split in f]
        parquet_files = [os.path.join(data_dir, f) for f in parquet_files if f.endswith('.parquet')]

        if not parquet_files:
            raise ValueError(f"No Parquet files found for split '{split}' in directory '{data_dir}'.")

        # Load dataset from local Parquet files
        dataset = load_dataset_hf('parquet', data_files=parquet_files, split="train")

        # Rename 'labels' column to 'label' if it exists
        if 'labels' in dataset.column_names:
            dataset = dataset.rename_column('labels', 'label')
            print("Renamed the column 'labels' to 'label'")

        return dataset

def load_trainer_state(checkpoint_path: str):
    """
    Loads the trainer state from a checkpoint file, for resuming training with the correct parameters.

    Args:
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        dict: The trainer_state.json as object, with the learning rate from the last log entry.
    """
    with open(checkpoint_path, 'r') as f:
        trainer_state = json.load(f)
    
    # we need to get the last learning rate for the optimizer from the log history. Not all entry contains a learning rate key, so we need to find the last one
    for entry in reversed(trainer_state['log_history']):
        if 'learning_rate' in entry:
            trainer_state['learning_rate'] = entry['learning_rate']
            break
    
    return trainer_state

def calculate_epochs(global_step: int, max_steps: int, dataset_size: int, batch_size: int, gradient_accumulation_steps: int, num_processes: int):
    """
    Calculates the number of epochs to train based on the global step, max steps, dataset size, batch size, gradient accumulation steps, and number of processes.

    Args:
        global_step (int): The current global step.
        max_steps (int): The maximum number of steps to train for.
        dataset_size (int): The size of the training dataset.
        batch_size (int): The batch size.
        gradient_accumulation_steps (int): The number of gradient accumulation steps.
        num_processes (int): The number of processes used for distributed training.

    Returns:
        float: The number of epochs to train for.
    """
    # Calculate the effective batch size
    effective_batch_size = batch_size * gradient_accumulation_steps * num_processes

    # Calculate the number of steps per epoch
    steps_per_epoch = dataset_size / effective_batch_size

    # Calculate the number of epochs
    num_epochs = (max_steps - global_step) / steps_per_epoch

    return num_epochs

def TrainerInit():
    if os.name == 'nt':
        try:    
            torch.distributed.init_process_group(backend="gloo")
        except:
            pass

    accelerator = Accelerator()

    parser = argparse.ArgumentParser(description='Train a model for tagging images')
    parser.add_argument('-c','--config', help='Path to the training config file', required=True)
    parser.add_argument('-w','--wandb', help='Use wandb for logging', action='store_true')
    parser.add_argument('-r','--resume', help='Resume training from the given checkpoint path', default=None)

    args = parser.parse_args()

    config = Config(args.config)

    device = accelerator.device

    return args, config, accelerator, device


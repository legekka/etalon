import os
import torch
from accelerate import Accelerator
from datasets import load_dataset as load_dataset_hf

from modules.config import Config

import argparse


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


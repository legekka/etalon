# HuCOLA: Evaluation metrics for language models measuring the grammatical correctness of hungarian sentences
# Models have to be finetuned as single_label_classification problems

import os
import argparse
import torch
import wandb
import random

from accelerate import Accelerator
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, get_scheduler, DataCollatorWithPadding, DataCollatorForLanguageModeling

from torch.optim import AdamW

from modules.config import Config
from modules.hulu import load_hucola_dataset
from modules.scheduler import CosineAnnealingWithWarmupAndEtaMin

import evaluate
import numpy as np

torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
random.seed(42)

accuracy = evaluate.load("accuracy")
matthews_corrcoef = evaluate.load("matthews_correlation")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    mcc = matthews_corrcoef.compute(predictions=predictions, references=labels)
    return {
        "accuracy": acc['accuracy'],
        "mcc": mcc['matthews_correlation']
    }

def tokenize_text(examples):
    global tokenizer
    global config
    inputs = tokenizer(
        examples['text'], 
        padding="max_length",
        truncation=True,
        max_length=config.max_seq_length,
        return_tensors="pt"
    )
    inputs["labels"] = examples["label"]
    return inputs

# if we are on windows, we need to check it, and set the torch backend to gloo
if os.name == 'nt':
    try:    
        torch.distributed.init_process_group(backend="gloo")
    except:
        pass

accelerator = Accelerator()
device = accelerator.device if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str, required=True, help='Path to the config file')
parser.add_argument('-w', '--wandb', action='store_true', help='Use wandb for logging')
parser.add_argument('-r', '--resume', type=str, default=None, help='Path to the checkpoint to resume training')
parser.add_argument('-s', '--sweep', action='store_true', help='Run a hyperparameter sweep')
parser.add_argument('--sweep_id', type=str, default=None, help='Sweep ID to resume')

args = parser.parse_args()

config = Config(args.config_path)

# Initialize the tokenizer
if args.resume is not None:
    tokenizer = AutoTokenizer.from_pretrained(args.resume, use_fast=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, use_fast=True)

if __name__ == '__main__':
    id2label = {0: "Incorrect", 1: "Correct"}
    label2id = {"Incorrect": 0, "Correct": 1}

    # Initialize the model
    if args.resume is not None:
        model = AutoModelForSequenceClassification.from_pretrained(args.resume, num_labels=2, id2label=id2label, label2id=label2id, problem_type="single_label_classification")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(config.model, num_labels=2, id2label=id2label, label2id=label2id, problem_type="single_label_classification")
    
    for param in model.parameters(): param.data = param.data.contiguous()
    
    model.to(device)
    torch.cuda.empty_cache()

    if accelerator.is_main_process:
        print(f"Model initialized with {model.num_parameters()} parameters.")
        print(f"Tokenizer initialized with {len(tokenizer)} tokens.")

    # Load the dataset
    dataset = load_hucola_dataset(config.train_dataset)
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    # shuffle the training dataset
    train_dataset = train_dataset.shuffle(seed=42)

    if accelerator.is_main_process:    
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")

    # global loss_fn
    # class_weights = get_class_weights(train_dataset)
    # loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    # if accelerator.is_main_process:
    #     print('Class weights calculated:', class_weights)

    # Apply tokenization on the fly
    train_dataset.set_transform(tokenize_text)
    eval_dataset.set_transform(tokenize_text)

    if config.num_epochs is None:
        num_epochs = config.max_steps * config.batch_size / len(train_dataset) * config.gradient_accumulation_steps * accelerator.num_processes
    else:
        num_epochs = config.num_epochs

    if accelerator.is_main_process:
        print("--- Hyperparameters ---")
        for key in config._jsonData.keys():
            print(f"{key}: {config._jsonData[key]}")
        print("-----------------------")
    
    # this is calculated based on the number of epochs and the length of the training dataset, divided by the (batch size * gradient accumulation steps * number of gpus)
    num_training_steps = num_epochs * len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps * accelerator.num_processes)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    if config.scheduler == "cosine" and config.eta_min != 0.0:
        scheduler = CosineAnnealingWithWarmupAndEtaMin(
            optimizer,
            T_max=num_training_steps,
            eta_min=config.eta_min,
            warmup_steps=config.warmup_steps
        )
    else:
        scheduler = get_scheduler(
            config.scheduler,
            optimizer=optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        remove_unused_columns=False,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        lr_scheduler_type=config.scheduler,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        logging_dir=config.output_dir,
        save_strategy="steps" if config.save_steps is not None else "epoch",
        save_steps=config.save_steps,
        eval_strategy="steps" if config.save_steps is not None else "epoch",
        eval_steps=config.save_steps,
        seed=4242,
        bf16=True if accelerator.mixed_precision == "bf16" else False,
        fp16=True if accelerator.mixed_precision == "fp16" else False,
        report_to="wandb" if args.wandb else "none",
        ddp_find_unused_parameters=False,
        dataloader_persistent_workers=True if config.num_workers > 0 else False,
        dataloader_num_workers=config.num_workers,
        warmup_steps=config.warmup_steps,
        # include_tokens_per_second=True,
        max_grad_norm=1.0
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = CustomTrainer(
        model=model,        
        args=training_args,
        #optimizers=(optimizer, scheduler),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[]
    )

    if args.wandb and accelerator.is_main_process:
        if args.sweep:
            # Sweep initialization
            sweep_config = {
                "name": "hubert-HuCOLA-sweep",
                "method": "bayes",
                "metric": {
                    "name": "eval/mcc",
                    "goal": "maximize"
                },
                "parameters": {
                    "batch_size": {
                        "values": [16, 64, 128, 256, 512]
                    },
                    "learning_rate": {
                        "values": [5e-6, 1e-5, 5e-5]
                    },
                    "max_steps": {
                        "values": [250, 500, 1000]
                    },
                }
            }

            # Initialize sweep
            if args.sweep_id is not None:
                sweep_id = args.sweep_id
            else:
                sweep_id = wandb.sweep(sweep_config, project=config.wandb["project"])

            def train_with_sweep():
                wandb.init()

                # Update parameters based on the sweep config
                batch_size = wandb.config.batch_size
                learning_rate = wandb.config.learning_rate
                max_steps = wandb.config.max_steps
                
                if batch_size > 64:
                    gradient_accumulation_steps = batch_size // 64
                    batch_size = 64
                else:
                    gradient_accumulation_steps = 1              

                if config.num_epochs is None:
                    num_epochs = max_steps * batch_size / len(train_dataset) * gradient_accumulation_steps * accelerator.num_processes
                else:
                    num_epochs = config.num_epochs

                max_steps = num_epochs * len(train_dataset) // (batch_size * gradient_accumulation_steps * accelerator.num_processes)

                eval_steps = max_steps // 5

                training_args = TrainingArguments(
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    learning_rate=learning_rate,
                    num_train_epochs=num_epochs,
                    output_dir=config.output_dir,
                    remove_unused_columns=False,
                    lr_scheduler_type=config.scheduler,
                    logging_steps=config.logging_steps,
                    logging_dir=config.output_dir,
                    save_strategy="steps",
                    save_steps=max_steps,
                    eval_strategy="steps",
                    eval_steps=eval_steps,
                    seed=4242,
                    bf16=True,
                    report_to="wandb",
                    ddp_find_unused_parameters=False,
                    dataloader_persistent_workers=True if config.num_workers > 0 else False,
                    dataloader_num_workers=config.num_workers,
                    warmup_steps=config.warmup_steps,
                    max_grad_norm=1.0
                )
                
                trainer = CustomTrainer(
                    model=model,        
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                )

                # Train with the current configuration
                trainer.train()

            # Run the sweep agent
            wandb.agent(sweep_id, train_with_sweep, project=config.wandb["project"])
        else:
            # Normal WandB run
            wandb.init(project=config.wandb["project"], name=config.wandb["name"], tags=config.wandb["tags"])
            wandb.config.update(config._jsonData)
            wandb.watch(model)


    model, optimizer, scheduler, train_dataset, eval_dataset, trainer = accelerator.prepare(
        model, optimizer, scheduler, train_dataset, eval_dataset, trainer
    )

    trainer.train()
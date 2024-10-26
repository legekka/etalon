import os
import argparse
import torch
import wandb
import random

from accelerate import Accelerator
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_scheduler, DataCollatorWithPadding
from datasets import load_dataset

from modules.config import Config
from modules.hulu import load_husst_dataset
from modules.scheduler import CosineAnnealingWithWarmupAndEtaMin

import evaluate
import numpy as np

torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
random.seed(42)

def get_class_weights(dataset):
    import numpy as np
    labels = dataset['label']
    class_counts = np.bincount(labels)

    class_weights = 1.0 / class_counts

    # normalize the weights
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    # convert to tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return class_weights

class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        global config
        # If DeepSpeed is enabled, no need to manually create optimizer or scheduler
        if self.args.deepspeed:
            return

        # Initialize the optimizer manually
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)

        # Initialize the scheduler manually
        if config.scheduler == "cosine" and config.eta_min != 0.0:
            self.lr_scheduler = CosineAnnealingWithWarmupAndEtaMin(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=config.eta_min,
                warmup_steps=self.args.warmup_steps
            )
        else:
            self.lr_scheduler = get_scheduler(
                config.scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        global loss_fn
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

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

args = parser.parse_args()

config = Config(args.config_path)

# Initialize the tokenizer
if args.resume is not None:
    tokenizer = AutoTokenizer.from_pretrained(args.resume, use_fast=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, use_fast=True)

if __name__ == '__main__':

    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label2id = {"negative": 0, "neutral": 1, "positive": 2}

    # Initialize the model
    if args.resume is not None:
        model = AutoModelForSequenceClassification.from_pretrained(args.resume, num_labels=3, id2label=id2label, label2id=label2id, problem_type="single_label_classification")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(config.model, num_labels=3, id2label=id2label, label2id=label2id, problem_type="single_label_classification")
    
    for param in model.parameters(): param.data = param.data.contiguous()
    
    model.to(device)
    torch.cuda.empty_cache()

    if accelerator.is_main_process:
        print(f"Model initialized with {model.num_parameters()} parameters.")
        print(f"Tokenizer initialized with {len(tokenizer)} tokens.")

    # Load the dataset
    dataset = load_husst_dataset(config.train_dataset)
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    # shuffle the training dataset
    train_dataset = train_dataset.shuffle(seed=42)

    if accelerator.is_main_process:    
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")

    global loss_fn
    class_weights = get_class_weights(train_dataset)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    if accelerator.is_main_process:
        print('Class weights calculated:', class_weights)

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
        bf16=True,
        report_to="wandb" if args.wandb else "none",
        ddp_find_unused_parameters=False,
        dataloader_persistent_workers=True if config.num_workers > 0 else False,
        dataloader_num_workers=config.num_workers,
        warmup_steps=config.warmup_steps,
        # include_tokens_per_second=True,
        max_grad_norm=1.0
    )

    # I don't know which datacollator to use here, ChatGPT
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
        wandb.init(project=config.wandb["project"], name=config.wandb["name"], tags=config.wandb["tags"])
        wandb.config.update(config._jsonData)
        wandb.watch(model)

    model.config.use_cache = False # mute warnings

    model, optimizer, scheduler, train_dataset, eval_dataset, trainer = accelerator.prepare(
        model, optimizer, scheduler, train_dataset, eval_dataset, trainer
    )
 
    trainer.train()
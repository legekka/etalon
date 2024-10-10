import torch
import wandb

from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers import AdamW, get_scheduler
from transformers import TrainingArguments, DefaultDataCollator

from modules.utils import load_dataset, TrainerInit, get_class_weights
from modules.metrics import F1
from modules.scheduler import CosineAnnealingWithWarmupAndEtaMin
from modules.custom_transforms import ImageClassificationTransform
from modules.custom_trainers import BalancedTrainer


def main():
    # Initializing Trainer Script
    args, config, accelerator, device = TrainerInit()

    image_processor = AutoImageProcessor.from_pretrained(config.model, use_fast=True)

    transforms = ImageClassificationTransform(image_processor, config.num_classes)

    # Loading the model
    if args.resume is not None:
        model = AutoModelForImageClassification.from_pretrained(args.resume)
    else:
        model = AutoModelForImageClassification.from_pretrained(config.model)
    
    model.to(device)

    if accelerator.is_main_process:
        print('Number of parameters:', model.num_parameters())
        print('Number of classes:', model.config.num_labels)

    # Creating datasets
    
    dataset = load_dataset(config.train_dataset, 'train')
    
    train_test_split = dataset.train_test_split(
        test_size=config.eval_dataset_split_size,
        train_size=len(dataset) - config.eval_dataset_split_size,
        shuffle=True,
        seed=42        
    )
    
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    if accelerator.is_main_process:    
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")

    # Calculating class weights

    class_weights = get_class_weights(train_dataset)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))

    if accelerator.is_main_process:
        print('Class weights calculated:', class_weights)

    # Setting up Trainer

    if config.num_epochs is None:
        num_epochs = (config.max_steps * config.batch_size * config.gradient_accumulation_steps * accelerator.num_processes) / len(train_dataset)
    else:
        num_epochs = config.num_epochs

    num_training_steps = num_epochs * len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps * accelerator.num_processes)
    
    # Optimizer and Scheduler

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    
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

    # Applying transforms

    train_dataset = train_dataset.set_transform(transforms.train)
    eval_dataset = eval_dataset.set_transform(transforms.evaluation)

    if accelerator.is_main_process:
        print('--- Hyperparameters ---')
        for key in config._jsonData.keys():
            print(f"{key}: {config._jsonData[key]}")
        print('-----------------------')

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
        save_steps=config.save_steps if config.save_steps is not None else None,
        eval_strategy="steps" if config.save_steps is not None or config.eval_steps is not None else "epoch",
        eval_steps=config.eval_steps if config.eval_steps is not None else config.save_steps if config.save_steps is not None else None,
        seed=4242,
        bf16=True,
        report_to="wandb" if args.wandb else "none",
        ddp_find_unused_parameters=False,
        dataloader_persistent_workers=False,
        dataloader_num_workers=config.num_workers,
        warmup_steps=config.warmup_steps,
        resume_from_checkpoint=args.resume if args.resume is not None else None,
    )

    data_collator = DefaultDataCollator()

    trainer = BalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=F1,
        callbacks=[],
    )

    # Applying Deepspeed Scheduler and Optimizer
    trainer.setup_custom_trainer_scheduler(scheduler, config.eta_min)
    
    # Applying class weights
    trainer.setup_custom_trainer_class_weights(loss_fn)

    if args.wandb and accelerator.is_main_process:
        wandb.init(project=config.wandb['project'], name=config.wandb['name'], tags=config.wandb['tags'])
        wandb.config.update(config._jsonData)
        wandb.watch(model)

    model.config.use_cache = False # mute warning

    model, optimizer, scheduler, train_dataset, eval_dataset = accelerator.prepare(
        model, optimizer, scheduler, train_dataset, eval_dataset
    )

    trainer.train()


if __name__ == '__main__':
    main()
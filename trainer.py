import wandb

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import TrainingArguments, DataCollatorForLanguageModeling, Trainer

from modules.utils import TrainerInit, load_trainer_state, calculate_epochs
from modules.custom_transforms import TokenizationTransform
from datasets import load_dataset


def main():
    # Initializing Trainer Script
    args, config, accelerator, device = TrainerInit()

    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

    transforms = TokenizationTransform(tokenizer, config.max_seq_length)

    # Loading the model
    if args.resume is not None:
        model = AutoModelForMaskedLM.from_pretrained(args.resume)
    else:
        model = AutoModelForMaskedLM.from_pretrained(config.model)
    
    model.to(device)

    if accelerator.is_main_process:
        print('Number of parameters:', model.num_parameters())
        print('Number of tokens:', len(tokenizer))

    # Creating datasets
    
    dataset = load_dataset(path=config.train_dataset["path"], name=config.train_dataset["name"], split=config.train_dataset["split"])
    
    train_test_split = dataset.train_test_split(
        test_size=config.eval_dataset_split_size,
        train_size=len(dataset) - config.eval_dataset_split_size,
        shuffle=True,
        seed=42        
    )
    
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    num_train_samples = len(train_dataset)

    if accelerator.is_main_process:    
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")

    # Applying transforms

    train_dataset.set_transform(transforms.tokenize_text)
    eval_dataset.set_transform(transforms.tokenize_text)

    # Setting up Trainer

    if config.num_epochs is None:
        num_epochs = calculate_epochs(0, config.max_steps, num_train_samples, config.batch_size, config.gradient_accumulation_steps, accelerator.num_processes)
    else:
        num_epochs = config.num_epochs

    save_strategy = "steps" if config.save_steps is not None else "epoch"
    save_steps = config.save_steps if config.save_steps is not None else None
    eval_strategy = "steps" if config.save_steps is not None or config.eval_steps is not None else "epoch"
    eval_steps = config.eval_steps if config.eval_steps is not None else config.save_steps if config.save_steps is not None else None
    bf16 = True if accelerator.mixed_precision == "bf16" else False
    fp16 = True if accelerator.mixed_precision == "fp16" else False

    if accelerator.is_main_process:
        print('--- Hyperparameters ---')
        for key in config._json_data.keys():
            print(f"{key}: {config._json_data[key]}")
        print('-----------------------')

    if args.resume is None:
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            remove_unused_columns=False,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            optim=config.optimizer,
            lr_scheduler_type=config.scheduler,
            learning_rate=config.learning_rate,
            logging_steps=config.logging_steps,
            logging_dir=config.output_dir,
            save_strategy=save_strategy,
            save_steps=save_steps,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            seed=4242,
            bf16=bf16,
            fp16=fp16,
            report_to="wandb" if args.wandb else "none",
            ddp_find_unused_parameters=False,
            dataloader_num_workers=config.num_workers,
            warmup_steps=config.warmup_steps,
        )
    else:
        trainer_state = load_trainer_state(args.resume)

        num_epochs = calculate_epochs(trainer_state["global_step"], config.max_steps, num_train_samples, config.batch_size, config.gradient_accumulation_steps, accelerator.num_processes)

        training_args = TrainingArguments(
            output_dir=config.output_dir,
            remove_unused_columns=False,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            learning_rate=trainer_state['learning_rate'],
            logging_steps=config.logging_steps,
            logging_dir=config.output_dir,
            save_strategy=save_strategy,
            save_steps=save_steps,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            seed=4242,
            bf16=bf16,
            fp16=fp16,
            report_to="wandb" if args.wandb else "none",
            ddp_find_unused_parameters=False,
            dataloader_num_workers=config.num_workers,
            resume_from_checkpoint=args.resume,
        )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=config.mlm_probability,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[],
    )

    if args.wandb and accelerator.is_main_process:
        wandb.init(project=config.wandb['project'], name=config.wandb['name'], tags=config.wandb['tags'])
        wandb.config.update(config._json_data)
        wandb.watch(model)

    trainer.create_optimizer()
    optimizer = trainer.optimizer

    model, optimizer, train_dataset, eval_dataset, trainer = accelerator.prepare(
        model, optimizer, train_dataset, eval_dataset, trainer
    )

    trainer.train()


if __name__ == '__main__':
    main()
import torch
from transformers import Trainer, AdamW, get_scheduler
from modules.scheduler import CosineAnnealingWithWarmupAndEtaMin
import numpy as np

class DeepspeedTrainer(Trainer):
    def setup_custom_trainer_scheduler(self, scheduler, eta_min=0.0):
        self.custom.scheduler = scheduler
        self.custom.eta_min = eta_min

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # If DeepSpeed is enabled, no need to manually create optimizer or scheduler
        if self.args.deepspeed:
            return

        # Initialize the optimizer manually
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0.01)

        # Initialize the scheduler manually
        if self.custom.scheduler == "cosine" and self.custom.eta_min != 0.0:
            self.lr_scheduler = CosineAnnealingWithWarmupAndEtaMin(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=self.custom.eta_min,
                warmup_steps=self.args.warmup_steps
            )
        else:
            self.lr_scheduler = get_scheduler(
                self.custom.scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps
            )

class BalancedTrainer(DeepspeedTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def setup_custom_trainer_class_weights(self, loss_fn):
        self.custom.loss_fn = loss_fn
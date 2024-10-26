from transformers import Trainer

class BalancedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def setup_custom_trainer_class_weights(self, loss_fn):
        self.custom.loss_fn = loss_fn
from torchvision import transforms as T
import numpy as np
import torch

class ImageClassificationTransform:
    def __init__(self, image_processor, num_classes):
        self.num_classes = num_classes
        self.size = (
            image_processor.size["shortest_edge"]
            if "shortest_edge" in image_processor.size
            else (image_processor.size["height"], image_processor.size["width"])
        )
        
        self._train_transforms = T.Compose([
            T.Resize(self.size),
            T.RandomAffine(
                degrees=15, 
                translate=(0.1, 0.1),
                scale=(0.75, 1.1),
                shear=None,
                fill=tuple(np.array(np.array(image_processor.image_mean) * 255).astype(int).tolist())
            ),
            T.ToTensor(),
            T.Normalize(
                mean=image_processor.image_mean, 
                std=image_processor.image_std
            )
        ])

        self._val_transforms = T.Compose([
            T.Resize(self.size),
            T.ToTensor(),
            T.Normalize(
                mean=image_processor.image_mean, 
                std=image_processor.image_std
            )
        ])

    def train(self, examples):
        inputs = {
            "pixel_values": [],
            "labels": []
        }

        inputs["pixel_values"] = [self._train_transforms(image.convert('RGB')) for image in examples["image"]]

        labels = []
        for label in examples["label"]:
            binary_vector = torch.zeros(self.num_classes)
            for l in label:
                binary_vector[l] = 1
            labels.append(binary_vector)
        inputs["labels"] = labels

        return inputs
    
    def evaluation(self, examples):
        inputs = {
            "pixel_values": [],
            "labels": []
        }

        inputs["pixel_values"] = [self._val_transforms(image.convert('RGB')) for image in examples["image"]]

        labels = []
        for label in examples["label"]:
            binary_vector = torch.zeros(self.num_classes)
            for l in label:
                binary_vector[l] = 1
            labels.append(binary_vector)
        inputs["labels"] = labels

        return inputs
    
class TokenizationTransform:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def tokenize_text(self, examples):
        inputs = self.tokenizer(
            examples["text"],
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt"
        )
        return inputs
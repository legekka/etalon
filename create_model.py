from transformers import ViTConfig, ViTForImageClassification
from modules.utils import get_label2id_id2label
import json

label2id, id2label = get_label2id_id2label()

config = ViTConfig(
    image_size=448,
    hidden_size=768,
    intermediate_size=3072,
    num_hidden_layers=12,
    num_attention_heads=12,
    patch_size=16,
    num_labels=len(label2id),
    layer_norm_eps=1e-12,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    classifier_dropout_prob=0.0,
    initializer_range=0.02,
    qkv_bias=True,
    hidden_act="gelu",
)

model = ViTForImageClassification(
    config
)

model.config.id2label = id2label
model.config.label2id = label2id
model.config.problem_type = "multi_label_classification"

preprocessor_config = {
    "do_normalize": True,
    "do_resize": True,
    "image_mean": [0.673, 0.608, 0.602],
    "image_std": [0.253, 0.250, 0.239],
    "size": 448,
}

# print parameter count 
print('Number of parameters:', model.num_parameters())

print(model)

# save model to disk
model.save_pretrained('./models/T12-ViT')
with open('./models/T12-ViT/preprocessor_config.json', 'w') as f:
    json.dump(preprocessor_config, f)
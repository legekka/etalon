from transformers import DebertaV2ForMaskedLM, DebertaV2Config, AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("tokenizer", use_fast=True)

config = DebertaV2Config(
    attention_probs_dropout_prob=0.1,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    hidden_size=1024,
    initializer_range=0.02,
    intermediate_size=4096,
    max_position_embeddings=2048,
    position_buckets=256,
    norm_rel_ebd="layer_norm",
    relative_attention=True,
    pos_att_type="p2c|c2p",
    layer_norm_eps=1e-7,
    max_relative_0s=-1,
    position_biased_input=False,
    share_att_key=True,
    num_attention_heads=16,
    num_hidden_layers=24,
    type_vocab_size=0,
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id
)

model = DebertaV2ForMaskedLM(config)
# print model paramter count
print(f"Model initialized with {model.num_parameters()} parameters.")
# print tokenizer vocab size
print(f"Tokenizer initialized with {len(tokenizer)} tokens.")

# model name: hu-deberta-v2-base
model.save_pretrained("models/hu-deberta-v2-large")
tokenizer.save_pretrained("models/hu-deberta-v2-large")
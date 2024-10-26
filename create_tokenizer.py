from transformers import BertTokenizerFast
import json
import os

# Paths to your files
vocab_file = "vocab.txt"  # Replace with your vocab.txt path
special_tokens_map_file = "special_tokens_map.json"  # Replace with your special_tokens_map.json path
tokenizer_save_dir = "tokenizer/"  # Replace with the directory where you want to save the tokenizer

# Load special tokens map
with open(special_tokens_map_file, "r", encoding="utf-8") as f:
    special_tokens_map = json.load(f)

# Initialize the tokenizer with the vocab file and special tokens map
tokenizer = BertTokenizerFast(
    vocab_file=vocab_file, 
    do_lower_case=False,
    **special_tokens_map)

# Save the tokenizer to a directory
os.makedirs(tokenizer_save_dir, exist_ok=True)
tokenizer.save_pretrained(tokenizer_save_dir)

print(f"Tokenizer saved successfully to {tokenizer_save_dir}")
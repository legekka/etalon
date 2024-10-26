import argparse

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Huggingface model name')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model.to(device)
    model.eval()

    text = "Anglia fővárosa [MASK]."
    print(f"Text: {text}")

    inputs = tokenizer(
        text,
        return_tensors="pt")
    # show token sequence with special tokens converted back to strings
    print(f"Tokens: {[tokenizer.convert_ids_to_tokens(token) for token in inputs['input_ids'][0].tolist()]}")
    # set the attention mask at the position of the mask token to 0
    print(f"Attention mask: {inputs['attention_mask'][0].tolist()}")
    inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # order the tokens by their probability, and get the top 5 with their probabilities
        probs = logits.softmax(dim=-1)[0, inputs['input_ids'][0].tolist().index(tokenizer.mask_token_id)].topk(5)
        for token, prob in zip(tokenizer.convert_ids_to_tokens(probs.indices), probs.values):
            print(f"{token}: {prob:.5f}")

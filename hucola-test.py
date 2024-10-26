from transformers import AutoModelForSequenceClassification, AutoTokenizer
from modules.hulu import load_hucola_dataset

import argparse
import json
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, help='Huggingface model')
    args = parser.parse_args()

    dataset = load_hucola_dataset("eval_datasets/HuCOLA")

    test_dataset = dataset["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    model.to(device)
    model.eval()

    results = []

    # get the longest tokenized sequence length
    print("Tokenizing the test dataset...")
    max_length = 0
    for i in range(len(test_dataset)):
        inputs = tokenizer(
            test_dataset["text"][i], 
            return_tensors="pt")
        max_length = max(max_length, inputs["input_ids"].shape[1])

    print(f"Max length: {max_length}")

    print("Generating predictions...")
    # generate the predictions for all the test dataset
    for i in range(len(test_dataset)):
        inputs = tokenizer(
            test_dataset["text"][i], 
            return_tensors="pt", 
            padding="max_length", 
            max_length=max_length,
            truncation=True)
        inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = logits.argmax(dim=-1)
            results.append(predictions.item())

    print("Saving results...")
    # create the output file
    datafile = []

    for i in range(len(test_dataset)):
        item = {
            "id": str(i),
            "label": str(results[i])
        }
        datafile.append(item)

    with open("result_hucola.json", "w", encoding="utf-8") as f:
        json.dump(datafile, f, ensure_ascii=False, indent=4)
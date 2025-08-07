from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

dataset = load_dataset(
    "json",
    data_files="https://huggingface.co/datasets/tasksource/figlang2020-sarcasm/resolve/main/sarcasm_detection_shared_task_reddit_training.jsonl",
    split="train"
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = []
labels = []
token_ids_list = []

for i, example in enumerate(dataset):
    if i >= 600:  
        break

    text = example["response"].strip()
    label = example["label"]

    if text == "":
        continue

    tagged_text = "<sarcasm>" + text
    token_ids = tokenizer(tagged_text, truncation=True, padding=False)["input_ids"]

    texts.append(tagged_text)
    labels.append(label)
    token_ids_list.append(" ".join(map(str, token_ids)))

    if (i + 1) % 100 == 0: 
        print(f"Processed {i+1} examples...") # This will indicate progress while the file runs.

df = pd.DataFrame({
    "text": texts,
    "label": labels,
    "tokens_str": token_ids_list
})

df.to_csv("final_clean_sarcasm.csv", index=False)
print("Saved to final_clean_sarcasm.csv")
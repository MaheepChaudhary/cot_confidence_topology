from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

dataset = load_dataset("truthful_qa", "generation")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
df = dataset["validation"].to_pandas()

def clean_and_tag(df):
    df = df.dropna(subset=['question'])
    df = df[df['question'].str.strip() != ""]

    df['question'] = df['question'].str.strip()

    df = df.drop_duplicates(subset=['question'])

    df['tagged_text'] = "<hallucination> " + df['question']

    return df

df_cleaned_tagged = clean_and_tag(df)

df_cleaned_tagged['tokens'] = df_cleaned_tagged['tagged_text'].apply(
    lambda x: tokenizer(x, truncation=True, padding="max_length", max_length=64)['input_ids']
)

df_cleaned_tagged['tokens_str'] = df_cleaned_tagged['tokens'].apply(lambda x: " ".join(map(str, x)))

columns_to_save = [
    "type",
    "category",
    "question",
    "best_answer",
    "correct_answers",
    "incorrect_answers",
    "source",
    "tagged_text",
    "tokens_str"
]

df_cleaned_tagged[columns_to_save].to_csv("final_clean_truthfulQA.csv", index=False)
print(df_cleaned_tagged[columns_to_save].head())

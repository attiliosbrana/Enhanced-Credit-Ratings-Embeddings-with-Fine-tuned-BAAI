import json
import os
import random
from dotenv import load_dotenv
from utils import estimate_api_usage
from query_generator import generate_queries
from corpus_loader import load_corpus
from prompt import prompt

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

all_files = [
    os.path.join("./data/pdfs", f)
    for f in os.listdir("./data/pdfs")
    if f.endswith(".pdf")
]
# select only 4% of the files
# all_files = random.sample(all_files, int(len(all_files) * 0.01))
random.seed(42)
random.shuffle(all_files)
split_index = int(0.8 * len(all_files))
TRAIN_FILES = all_files[:split_index]
VAL_FILES = all_files[split_index:]

# Load the Corpus
train_corpus = load_corpus(TRAIN_FILES, verbose=True)
val_corpus = load_corpus(VAL_FILES, verbose=True)

# Estimate API calls and tokens
(
    total_input_tokens_train,
    total_output_tokens_train,
    api_calls_train,
) = estimate_api_usage(train_corpus, prompt_template=prompt)
total_input_tokens_val, total_output_tokens_val, api_calls_val = estimate_api_usage(
    val_corpus, prompt_template=prompt
)

total_input_tokens = total_input_tokens_train + total_input_tokens_val
total_output_tokens = total_output_tokens_train + total_output_tokens_val
total_api_calls = api_calls_train + api_calls_val

print(f"Estimated API calls: {total_api_calls}")
print(f"Estimated input tokens: {total_input_tokens}")
print(f"Estimated output tokens: {total_output_tokens}")

# Cost estimation
estimated_input_cost = (total_input_tokens / 1000) * 0.0015
estimated_output_cost = (total_output_tokens / 1000) * 0.002

total_estimated_cost = estimated_input_cost + estimated_output_cost
print(f"Estimated cost: ${total_estimated_cost:.4f}")

# Generate and Save Queries
train_queries, train_relevant_docs = generate_queries(
    train_corpus, prompt_template=prompt, openai_api_key=openai_api_key
)
val_queries, val_relevant_docs = generate_queries(
    val_corpus, prompt_template=prompt, openai_api_key=openai_api_key
)

with open("./data/train_queries.json", "w+") as f:
    json.dump(train_queries, f)
with open("./data/train_relevant_docs.json", "w+") as f:
    json.dump(train_relevant_docs, f)
with open("./data/val_queries.json", "w+") as f:
    json.dump(val_queries, f)
with open("./data/val_relevant_docs.json", "w+") as f:
    json.dump(val_relevant_docs, f)

# Merge and Save Data
train_dataset = {
    "queries": train_queries,
    "corpus": train_corpus,
    "relevant_docs": train_relevant_docs,
}

val_dataset = {
    "queries": val_queries,
    "corpus": val_corpus,
    "relevant_docs": val_relevant_docs,
}

with open("./data/train_dataset.json", "w+") as f:
    json.dump(train_dataset, f)
with open("./data/val_dataset.json", "w+") as f:
    json.dump(val_dataset, f)

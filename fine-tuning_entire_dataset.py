# Import required packages
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
import json

# Load pretrained model
model_id = "BAAI/bge-large-en"
model = SentenceTransformer(model_id)

# Define paths to training and validation data
TRAIN_DATASET_FPATH = "./data/train_dataset.json"
VAL_DATASET_FPATH = "./data/val_dataset.json"

# Load training and validation datasets
with open(TRAIN_DATASET_FPATH, "r") as f:
    train_dataset = json.load(f)

with open(VAL_DATASET_FPATH, "r") as f:
    val_dataset = json.load(f)

# Merge train and validation datasets
merged_corpus = {**train_dataset["corpus"], **val_dataset["corpus"]}
merged_queries = {**train_dataset["queries"], **val_dataset["queries"]}
merged_relevant_docs = {
    **train_dataset["relevant_docs"],
    **val_dataset["relevant_docs"],
}

# Define batch size
BATCH_SIZE = 8

# Prepare merged training data
merged_examples = []
for query_id, query in merged_queries.items():
    node_id = merged_relevant_docs[query_id][0]
    text = merged_corpus[node_id]
    example = InputExample(texts=[query, text])
    merged_examples.append(example)

# Pretty print merged_examples
print("Number of merged_examples: ", len(merged_examples))
# Create DataLoader from merged examples
merged_loader = DataLoader(merged_examples, batch_size=BATCH_SIZE)

# Define loss
loss = losses.MultipleNegativesRankingLoss(model)

# Training settings
EPOCHS = 2
warmup_steps = int(len(merged_loader) * EPOCHS * 0.1)

# Run training on the entire dataset
model.fit(
    train_objectives=[(merged_loader, loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path="exp_finetune_entire_dataset",
    show_progress_bar=True,
)

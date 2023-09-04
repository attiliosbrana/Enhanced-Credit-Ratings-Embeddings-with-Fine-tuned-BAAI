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

# Define batch size
BATCH_SIZE = 8

# Prepare training data
train_corpus = train_dataset["corpus"]
train_queries = train_dataset["queries"]
train_relevant_docs = train_dataset["relevant_docs"]

train_examples = []
for query_id, query in train_queries.items():
    node_id = train_relevant_docs[query_id][0]
    text = train_corpus[node_id]
    example = InputExample(texts=[query, text])
    train_examples.append(example)

train_loader = DataLoader(train_examples, batch_size=BATCH_SIZE)

# Prepare validation data
val_corpus = val_dataset["corpus"]
val_queries = val_dataset["queries"]
val_relevant_docs = val_dataset["relevant_docs"]

# Set up evaluator
evaluator = InformationRetrievalEvaluator(val_queries, val_corpus, val_relevant_docs)

# Define loss
loss = losses.MultipleNegativesRankingLoss(model)

# Training settings
EPOCHS = 2
warmup_steps = int(len(train_loader) * EPOCHS * 0.1)

# Run training
model.fit(
    train_objectives=[(train_loader, loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path="exp_finetune_optimal",
    show_progress_bar=True,
    evaluator=evaluator,
    evaluation_steps=50,
)

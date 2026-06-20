from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch

# 1. Load the dataset
dataset = load_dataset("imdb")

# 2. Preprocessing: Tokenization
# We'll use a standard fast tokenizer (like BERT's) to handle the text
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Apply tokenization to the whole dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Format for PyTorch
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# 4. Create DataLoaders
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=32)
test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=32)

from model import S4Model
import torch
from dataset import tokenizer,train_dataloader
import torch.nn as nn
import torch.optim as optim
from train import train
from inference import predict_sentiment

# Define model parameters
# input_dim was previously 10, but now we use vocab_size for embedding
d_model = 64        # Internal feature dimension for S4 layers and embedding dimension
state_dim = 128     # Internal state dimension (N) for S4 layers
num_classes = 2     # Number of output classes for classification (IMDB has 2 classes: pos/neg)
num_layers = 2       # Number of stacked S4 layers
L_max = 512         # Max sequence length for kernel computation

# Get vocab size from the tokenizer (defined in cell XmiSEUsz2sFf)
vocab_size = tokenizer.vocab_size

# Instantiate the S4Model
model = S4Model(vocab_size=vocab_size, # Use vocab_size here
                   d_model=d_model,
                   state_dim=state_dim,
                   num_classes=num_classes,
                   num_layers=num_layers,
                   L_max=L_max)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

num_epochs = 5

train(num_epochs,model,train_dataloader,optimizer,criterion,device)

print(predict_sentiment("This movie was an absolutly fantastic!", model, tokenizer, device))
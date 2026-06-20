import torch.optim as optim
from model import S5Model
import torch
import torch.nn as nn
from train import train

# --- Training Configuration ---
learning_rate = 0.001
num_epochs = 50

# using the same parameters as the test cell
batch_size = 4
seq_len = 100
input_dim = 10
output_dim = 5
hidden_dim = 20
state_dim = 32
num_layers = 2

s5_model = S5Model(num_layers, state_dim, input_dim, hidden_dim, output_dim)

# --- Loss Function and Optimizer ---
# For a sequence-to-sequence regression task, MSE is a common choice.
criterion = nn.MSELoss()
optimizer = optim.Adam(s5_model.parameters(), lr=learning_rate)

# --- Generate Synthetic Training Data (Input and Target) ---
# We'll use the same synthetic_input from the test, and create a synthetic target
# In a real scenario, `synthetic_input` would be your training features and `synthetic_target` your labels.
synthetic_input_train = torch.randn(batch_size, seq_len, input_dim)
synthetic_target = torch.randn(batch_size, seq_len, output_dim) # Target for regression

train(num_epochs,synthetic_input_train,synthetic_target,s5_model,optimizer,criterion)

# --- Generate new synthetic data for inference ---
# The input dimensions should match what the model was trained on
# batch_size, seq_len, input_dim are already defined from previous cells
synthetic_input_inference = torch.randn(batch_size, seq_len, input_dim)

print(f"Synthetic input for inference shape: {synthetic_input_inference.shape}")

# --- Perform Inference ---
s5_model.eval() # Set the model to evaluation mode

with torch.no_grad(): # Disable gradient calculation for inference
    inference_output = s5_model(synthetic_input_inference)

print(f"\nInference output shape: {inference_output.shape}")
print("\nFirst batch, first 5 time steps, first 3 output dimensions of inference output:")
print(inference_output[0, :5, :3])
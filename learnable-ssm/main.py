from dataset import simulate_pytorch_ssm
import matplotlib.pyplot as plt
import torch
from model import LearnableSSM
import torch.optim as optim
from train import train
from losses import calculate_nll_loss

# --- True Model Parameters for Data Generation ---
true_state_dim = 2
true_obs_dim = 1

true_A = torch.tensor([[1., 1.], [0., 1.]]) # Constant velocity model
true_Q = torch.diag(torch.tensor([0.01, 0.001])) # Small process noise
true_C = torch.tensor([[1., 0.]]) # Observe position only
true_R = torch.diag(torch.tensor([0.5])) # Moderate observation noise
true_x0 = torch.tensor([[0.], [1.]]) # Initial position 0, velocity 1

num_steps = 100

true_states, observed_data = simulate_pytorch_ssm(true_A, true_C, true_Q, true_R, true_x0, num_steps)

# Plot the simulated data
plt.figure(figsize=(12, 6))
plt.plot(true_states[:, 0].numpy(), label='True Position', color='blue')
plt.scatter(range(num_steps), observed_data[:, 0].numpy(), label='Observed Position', color='red', marker='x', s=20, alpha=0.7)
plt.title('Simulated 2D State Space Model Data (Position)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# --- Instantiate and Run the Learnable SSM ---
state_dim = 2
obs_dim = 1

# Initialize the model
ssm_model = LearnableSSM(state_dim, obs_dim)

# Run the forward pass (Kalman Filter) with the simulated observations
# Note: At this point, the parameters are randomly initialized, so the estimate
# won't be good yet. Training would adjust these parameters.
estimated_states, estimated_covariances, _, _ = ssm_model(observed_data) # Corrected to unpack all four return values

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(true_states[:, 0].numpy(), label='True Position', color='blue')
plt.plot(true_states[:, 1].numpy(), label='True Velocity', color='purple', linestyle=':')
plt.scatter(range(num_steps), observed_data[:, 0].numpy(), label='Observed Position', color='red', marker='x', s=20, alpha=0.7)
plt.plot(estimated_states[:, 0].detach().numpy(), label='Estimated Position', color='green', linestyle='--')
plt.plot(estimated_states[:, 1].detach().numpy(), label='Estimated Velocity', color='orange', linestyle='--')

plt.title('True, Observed, and (Initial) Estimated States from Learnable SSM')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

print("\nInitial Learnable Parameters (before training):")
print(f"A:\n{ssm_model.A.data}")
print(f"Q_diag (softplus applied):\n{torch.nn.functional.softplus(ssm_model.Q_diag).data}")
print(f"C:\n{ssm_model.C.data}")
print(f"R_diag (softplus applied):\n{torch.nn.functional.softplus(ssm_model.R_diag).data}")
print(f"initial_x:\n{ssm_model.initial_x.data}")
print(f"initial_P_diag (softplus applied):\n{torch.nn.functional.softplus(ssm_model.initial_P_diag).data}")

criterion=calculate_nll_loss
# Optimizer
optimizer = optim.Adam(ssm_model.parameters(), lr=0.01)

num_epochs = 200 # You can adjust the number of epochs

train(num_epochs,observed_data,ssm_model,optimizer,criterion,obs_dim)

# --- Plot results after training ---
estimated_states_trained, estimated_covariances_trained, _, _ = ssm_model(observed_data)

plt.figure(figsize=(12, 6))
plt.plot(true_states[:, 0].numpy(), label='True Position', color='blue')
plt.plot(true_states[:, 1].numpy(), label='True Velocity', color='purple', linestyle=':')
plt.scatter(range(num_steps), observed_data[:, 0].numpy(), label='Observed Position', color='red', marker='x', s=20, alpha=0.7)
plt.plot(estimated_states_trained[:, 0].detach().numpy(), label='Estimated Position (Trained)', color='green', linestyle='--')
plt.plot(estimated_states_trained[:, 1].detach().numpy(), label='Estimated Velocity (Trained)', color='orange', linestyle='--')

plt.title('True, Observed, and Estimated States from Learnable SSM (After Training)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

print("\nLearned Parameters (after training):")
print(f"A:\n{ssm_model.A.data}")
print(f"Q_diag (softplus applied):\n{torch.nn.functional.softplus(ssm_model.Q_diag).data}")
print(f"C:\n{ssm_model.C.data}")
print(f"R_diag (softplus applied):\n{torch.nn.functional.softplus(ssm_model.R_diag).data}")
print(f"initial_x:\n{ssm_model.initial_x.data}")
print(f"initial_P_diag (softplus applied):\n{torch.nn.functional.softplus(ssm_model.initial_P_diag).data}")
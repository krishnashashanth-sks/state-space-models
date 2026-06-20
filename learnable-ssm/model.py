import torch
import torch.nn as nn
torch.manual_seed(42)

class LearnableSSM(nn.Module):
  def __init__(self,state_dim,obs_dim):
    super(LearnableSSM,self).__init__()
    self.state_dim=state_dim
    self.obs_dim=obs_dim
    self.A=nn.Parameter(torch.eye(state_dim))
    if state_dim==2 and obs_dim==1 :
      self.A.data=torch.Tensor([[1.,1.],[0.,1.]])
    self.Q_diag=nn.Parameter(torch.log(torch.exp(torch.ones(state_dim)*0.1)-1.))
    self.C=nn.Parameter(torch.randn(obs_dim,state_dim))
    if state_dim==2 and obs_dim==1: # Changed condition to match state_dim=2
      self.C.data=torch.Tensor([[1.,0.]])
    self.R_diag=nn.Parameter(torch.log(torch.exp(torch.ones(obs_dim)*0.5)-1.))
    self.initial_x=nn.Parameter(torch.zeros(state_dim,1)) # Corrected typo initial_z to initial_x
    self.initial_P_diag=nn.Parameter(torch.log(torch.exp(torch.ones(state_dim)*1.0)-1.)) # Adjusted initialization for initial_P_diag
  def forward(self,observations): # Corrected 'observation' to 'observations' to match usage
    Q=torch.diag(torch.nn.functional.softplus(self.Q_diag))
    R=torch.diag(torch.nn.functional.softplus(self.R_diag))
    P_0=torch.diag(torch.nn.functional.softplus(self.initial_P_diag))
    num_steps=observations.shape[0]
    estimated_states=torch.zeros(num_steps,self.state_dim)
    estimated_covariances=torch.zeros(num_steps,self.state_dim,self.state_dim)

    # Store innovations and innovation covariances for NLL calculation
    innovations_list = []
    innovation_covariances_list = []

    x_t=self.initial_x.clone() # Corrected typo initiali_z to initial_x
    P_t=P_0.clone()
    I=torch.eye(self.state_dim)
    for t in range(num_steps):
      x_pred=self.A @ x_t
      P_pred=self.A @ P_t @ self.A.T + Q # Corrected missing '@' operator
      y_t_obs=observations[t,:].reshape(self.obs_dim,1)
      innovation=y_t_obs-(self.C @ x_pred)
      innovation_covariance=self.C @ P_pred @ self.C.T + R

      innovations_list.append(innovation)
      innovation_covariances_list.append(innovation_covariance)

      try:
        S_inv=torch.linalg.inv(innovation_covariance+torch.eye(self.obs_dim)*1e-6)
      except torch.linalg.LinAlgError:
        print(f"Warning: Singular matrix at time step {t}. Adding regularization.")
        S_inv=torch.linalg.inv(innovation_covariance+torch.eye(self.obs_dim)*1e-4)
      Kalman_gain=P_pred @ self.C.T @ S_inv # Corrected typo kal_mangain
      x_t=x_pred + Kalman_gain @ innovation # Corrected state update equation
      P_t=(I-Kalman_gain @ self.C)@ P_pred # Corrected typo Kalma_gain
      estimated_states[t,:]=x_t.flatten()
      estimated_covariances[t,:,:]=P_t
    return estimated_states,estimated_covariances, innovations_list, innovation_covariances_list 
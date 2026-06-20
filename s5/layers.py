import torch
import torch.nn as nn
import torch.nn.functional as F
import math # Import math module for log

class S5Layer(nn.Module):
  def __init__(self,state_dim,input_dim,output_dim,discretization='zoh',dt_min=0.001,dt_max=0.1):
    super().__init__()
    self.state_dim=state_dim
    self.input_dim=input_dim
    self.output_dim=output_dim
    self.discretization=discretization
    self.A_bar=nn.Parameter(torch.randn(state_dim,state_dim))
    self.B_bar=nn.Parameter(torch.randn(state_dim,input_dim))
    self.C_bar=nn.Parameter(torch.randn(output_dim,state_dim))
    self.log_dt=nn.Parameter(torch.rand(1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))

  def discretize(self):
    dt=torch.exp(self.log_dt)
    I=torch.eye(self.state_dim,device=self.A_bar.device)
    A_discrete=I+dt*self.A_bar+0.5*(dt**2)*(self.A_bar @ self.A_bar) # Fixed: Changed self.B_bar to self.A_bar
    B_discrete=dt*self.B_bar+0.5*(dt**2)*(self.A_bar @ self.B_bar)
    return A_discrete,B_discrete

  def forward(self,u):
    batch_size,seq_len,_=u.shape
    A_discrete,B_discrete=self.discretize()
    C_discrete=self.C_bar
    h=torch.zeros(batch_size,self.state_dim,device=u.device)
    output=[]
    for t in range(seq_len):
      u_t=u[:,t,:].unsqueeze(-1)
      h=A_discrete @ h.unsqueeze(-1)+B_discrete @ u_t
      h=h.squeeze(-1)
      y_t=C_discrete @ h.unsqueeze(-1)
      output.append(y_t.squeeze(-1))
    return torch.stack(output,dim=1)
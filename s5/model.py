from layers import S5Layer
import torch.nn as nn

class S5Model(nn.Module):
  def __init__(self,num_layers,state_dim,input_dim,hidden_dim,output_dim):
    super().__init__()
    self.input_proj=nn.Linear(input_dim,hidden_dim)
    self.s5_layers=nn.ModuleList([
        S5Layer(state_dim,hidden_dim,hidden_dim)for _ in range(num_layers)
    ])
    self.output_proj=nn.Linear(hidden_dim,output_dim)
  def forward(self,x):
    x=self.input_proj(x)
    for layer in self.s5_layers:
      x=layer(x)
    return self.output_proj(x)

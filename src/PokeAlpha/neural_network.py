import torch
import torch.nn as nn

class neural_network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU()
        )

        self.policy_head=nn.Linear(128, output_dim)
        self.value_head=nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Tanh()
        )
    def policy_forward(self,x):
        x=self.main(x)
        return self.policy_head(x)
    
    def value_forward(self,x):
        x=self.main(x)
        return self.value_head(x)

    def forward(self,x):
        return self.policy_forward(x),self.value_forward(x)


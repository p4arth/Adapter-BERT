import torch
from config import *
from torch import nn
import torch.nn.functional as F

class AdapterBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.project_down = nn.Linear(in_features = in_features,
                                      out_features = ADAPTER_BOTTLENECK)
        self.project_up = nn.Linear(in_features = ADAPTER_BOTTLENECK, 
                                    out_features = in_features)
    
    def forward(self, x):
        x_clone = x.clone()
        x = self.project_down(x)
        x = F.relu(x)
        x = self.project_up(x) + x_clone
        return x
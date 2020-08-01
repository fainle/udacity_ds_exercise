import torch
import torch.nn.functional as F

from torch import nn


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        hidder_size = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in hidder_size])
         
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        for e in self.hidden_layers:
            x = F.relu(e(x))
            x = self.dropout(x)
        
        x = self.output(x)
        return F.log_softmax(x)
    

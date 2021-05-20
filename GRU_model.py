import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x will be of shape: [batch size, sequence length, input size  which will be one]
        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.gru(x, hidden) # GRU model requires shape of: [batch size, sequence length]
        # output size will be [batch size, sequence length, hidden size]
        
        out2 = out[:, -1, :] # we will take the last sample from hidden size
        
        return torch.sigmoid(self.fc(out2))




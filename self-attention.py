import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadSelfAttention (nn.Module):
    def __init__(self, E_d, H_d):
        super().__init__()
        
        self.Wq = nn.Linear(E_d,H_d)
        self.Wk = nn.Linear(E_d,H_d)
        self.Wv = nn.Linear(E_d,H_d)
        
        self.scale = H_d ** 0.5
        
    def forward(self, x):
        
        Q = self.Wq(x)
        
        K = self.Wk(x)
        
        V = self.Wv(x)
        
        attention_score = (Q @ K.transpose(-2,-1))/ self.scale
        
        attention_weights = F.softmax(attention_score, -1)
        
        attention_output = attention_weights @ V
        
        return attention_output
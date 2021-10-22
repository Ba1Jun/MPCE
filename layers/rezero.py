import torch.nn as nn
import torch

class RezeroConnection(nn.Module):
    def __init__(self):
        super(RezeroConnection, self).__init__()
        self.rezero_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float), requires_grad=True)

    def forward(self, x, sublayer):
        return x + self.rezero_alpha * (sublayer(x))
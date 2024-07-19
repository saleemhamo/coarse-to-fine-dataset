import torch
import torch.nn as nn

class SupervisionLoss(nn.Module):
    def __init__(self):
        super(SupervisionLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, predicted, target):
        return self.loss(predicted, target)

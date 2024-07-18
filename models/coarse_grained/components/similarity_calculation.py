import torch


class SimilarityCalculation(torch.nn.Module):
    def __init__(self, dim):
        super(SimilarityCalculation, self).__init__()
        self.fc = torch.nn.Linear(dim, 1)

    def forward(self, aligned_emb):
        return self.fc(aligned_emb)

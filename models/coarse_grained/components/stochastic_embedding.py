import torch


class StochasticEmbeddingModule(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StochasticEmbeddingModule, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)
        self.noise_std = 0.1

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_std
        return self.fc(x) + noise

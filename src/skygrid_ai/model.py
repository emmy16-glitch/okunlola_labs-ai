
import torch
from torch import nn


class SkyGridTelemetryNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Two inputs, sixteen hidden neurons
        self.hidden = nn.Linear(2, 16)

        # Sixteen hidden outputs, one final output
        self.output = nn.Linear(16, 1)

    def forward(self, inputs):
        x = torch.relu(self.hidden(inputs))
        return self.output(x)

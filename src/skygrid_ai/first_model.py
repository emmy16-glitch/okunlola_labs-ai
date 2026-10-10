
import torch
from torch import nn
from pathlib import Path

# Make experiments repeatable.
torch.manual_seed(42)

# Generate artificial training examples.
def make_data(count):
    inputs = torch.rand(count, 2)

    # Synthetic labels only, NOT real flight-safety rules.
    battery = inputs[:, 0]
    gps = inputs[:, 1]

    labels = ((battery < 0.20) | (gps < 0.30))
    labels = labels.float().unsqueeze(1)

    return inputs, labels


train_x, train_y = make_data(400)
test_x, test_y = make_data(100)


# Our first custom neural network architecture.
class SkyGridTelemetryNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(2, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, inputs):
        x = self.hidden(inputs)
        x = torch.relu(x)
        return self.output(x)


model = SkyGridTelemetryNet()

loss_function = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.02
)

# Train our network from random initial weights.
for epoch in range(301):
    model.train()

    prediction = model(train_x)
    loss = loss_function(prediction, train_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")


# Evaluate using examples not used during training.
model.eval()

with torch.no_grad():
    logits = model(test_x)
    predictions = (torch.sigmoid(logits) >= 0.5).float()
    accuracy = (predictions == test_y).float().mean()

    print(f"Test accuracy: {accuracy.item() * 100:.2f}%")

    # Example: 12% battery, 16/20 GPS satellites.
    sample = torch.tensor([[0.12, 0.80]])
    probability = torch.sigmoid(model(sample)).item()

    print(f"Simulated attention score: {probability:.4f}")


# Save the neural network weights.
Path("checkpoints").mkdir(exist_ok=True)

torch.save(
    model.state_dict(),
    "checkpoints/skygrid_telemetry_v01.pt"
)

print("SkyGrid model saved successfully!")

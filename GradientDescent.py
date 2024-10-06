import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

x = torch.tensor([2.0])
y = torch.tensor([4.0])


class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.w12 = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        return self.w12 * x


model = SimpleLinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 50
loss_values = []
weight_history = []
predicted_values = []

model.w12.data = torch.tensor([0.0])

for epoch in range(epochs):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_values.append(loss.item())
    weight_history.append(model.w12.item())
    predicted_values.append(y_pred.item())

    print(
        f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Predicted Output: {y_pred.item():.4f}"
    )

weight_range = np.linspace(-2, 6, 100)
loss_surface = []

for w in weight_range:
    model.w12.data = torch.tensor([w])
    with torch.no_grad():
        y_pred_surface = model(x)
        loss_surface.append(criterion(y_pred_surface, y).item())

loss_surface = np.array(loss_surface)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(weight_range, loss_surface, label="Loss Surface", color="blue")
plt.plot(
    weight_history,
    [criterion(torch.tensor([w]) * x, y).item() for w in weight_history],
    "ro-",
    label="Gradient Descent Path",
)
plt.title("Loss Surface with Gradient Descent Path")
plt.xlabel("Weight (w12)")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(range(epochs), loss_values, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over Epochs in Gradient Descent")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(range(epochs), predicted_values, label="Predicted Output", color="orange")
plt.axhline(y=4.0, color="red", linestyle="--", label="Actual Output (y=4)")
plt.xlabel("Epochs")
plt.ylabel("Output")
plt.title("Predicted Output Approaching Actual Output")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

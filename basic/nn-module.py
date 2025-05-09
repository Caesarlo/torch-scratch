import torch
import math

from caesar.logger.logger import setup_logger

logger = setup_logger(__name__)

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)


model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

loss_fn = torch.nn.MSELoss(reduction="sum")

learnig_rate = 1e-6

for t in range(2000):
    y_pred = model(xx)

    loss = loss_fn(y_pred, y)

    if t % 100 == 99:
        logger.info(f"Step: {t} Loss: {loss}")

    model.zero_grad()

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learnig_rate*param.grad


linear_layer = model[0]

logger.info(
    f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

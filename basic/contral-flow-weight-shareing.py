import random
import torch
import math

from caesar.logger.logger import setup_logger

logger = setup_logger(__name__)


class DynamicNet(torch.nn.Module):
    def __init__(self):
        super(DynamicNet, self).__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.a+self.b*x+self.c*x**2+self.d*x**3
        for exp in range(4, random.randint(4, 6)):
            y = y+self.e*x**exp
        return y

    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 ? + {self.e.item()} x^5 ?'


x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)


model = DynamicNet()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)

for t in range(30000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if t % 2000 == 1999:
        logger.info(f"Step: {t} Loss: {loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

logger.info(f"Result: {model.string()}")

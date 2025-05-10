import torch
import math

from caesar.logger.logger import setup_logger

logger = setup_logger(__name__)


class Polynomial3(torch.nn.Module):
    def __init__(self):
        super(Polynomial3, self).__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a+self.b*x+self.c*x**2+self.d*x**3

    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'


x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

model = Polynomial3()


criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

for t in range(2000):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    if t % 100 == 99:
        logger.info(f'Step: {t} Loss: {loss.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


logger.info(f"Result: {model.string()}")

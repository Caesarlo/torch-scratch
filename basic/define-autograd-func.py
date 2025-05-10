import torch
import math

from caesar.logger.logger import setup_logger

logger = setup_logger(__name__)


class LegendrePolynomial3(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 0.5*(5*input**3-3*input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output*1.5*(5*input**2-1)


dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

learning_rate = 5e-6
for t in range(2000):
    P3 = LegendrePolynomial3.apply

    y_pred = a+b*P3(c+d*x)

    loss = (y_pred-y).pow(2).sum()

    if t % 100 == 99:
        logger.info(f"Step: {t} Loss {loss}")

    loss.backward()

    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None


logger.info(
    f'Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)')

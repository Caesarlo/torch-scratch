import numpy as np
import math

from caesar.logger.logger import setup_logger


logger = setup_logger(__name__)

x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)


a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()


learnig_rate = 1e-6


for t in range(2000):
    y_pred = a+b*x+c*x**2+d*x**3

    loss = np.square(y_pred-y).sum()
    if t % 100 == 99:
        logger.info(f'Step {t}, Loss: {loss}')

    grad_y_pred = 2.0*(y_pred-y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred*x).sum()
    grad_c = (grad_y_pred*x**2).sum()
    grad_d = (grad_y_pred*x**3).sum()

    a -= learnig_rate*grad_a
    b -= learnig_rate*grad_b
    c -= learnig_rate*grad_c
    d -= learnig_rate*grad_d

logger.info(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')

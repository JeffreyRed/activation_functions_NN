import numpy as np
import matplotlib.pyplot as plt

from activation_functions import (
    gelu,
    sigmoid,
    softmax,
    tanh,
    relu,
    leaky_relu,
    elu,
    swish,
)

x = np.linspace(-10, 10, 1000)

functions = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "elu": elu,
    "swish": swish,
    "softmax": softmax,
    "gelu": gelu,
}

for name, func in functions.items():
    plt.figure(figsize=(8, 5))
    plt.plot(x, func(x), linewidth=2)
    plt.title(f"{name.upper()} Activation Function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.savefig(f"images/{name}.png")
    plt.close()
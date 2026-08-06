# Testing saving of the model
# and loading

import json
import os
import sys

# to be able to import base dir modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from micrograd import nn
from micrograd.engine import Value


# Debug: print tree like structure from parent to child
# for small debugging
def log_root(root, level=0, offset=5):
    indent = " " * level * offset
    print(f"{indent}|__" if level else "", root)

    if root._prev:
        for child in root._prev:
            log_root(child, level + 1)


def test_val():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f

    log_root(g)


def save_neuron():
    n = nn.Neuron(2)
    x = [Value(1.0), Value(-2.0)]
    y = n(x)

    y.backward()
    with open("test.json", "w", encoding="utf-8") as f:
        json.dump(n._data(), f, indent=2)
    print(n)


def load_neuron():
    with open("test.json", "r", encoding="utf-8") as f:
        data = json.load(f)

        n = nn.Neuron.data_(data)
        return n


if __name__ == "__main__":
    FILENAME = "test/test.json"
    epochs = 1000
    lr = 0.01

    X = [
        [2.4, 5.0, 7.0, 8.8],
        [4.1, 1.1, 5.7, 7, 8],
        [2.1, 1.7, 2.7, 1, 8],
        [1.8, 0.1, 2.3, 7, 8],
    ]
    y = [1, 4, 2, 4]

    # re improve existing model
    if not os.path.isfile(FILENAME):
        nn = nn.MLP(4, [4, 4, 1])
        print(f"Initial:{[nn(x).data for x in X]}")
        nn.fit(X, y, lr=lr, epochs=epochs)
        nn.save(FILENAME)
        print(f"After:{[nn(x).data for x in X]}")
        print(f"Actual:{y}")
    else:
        nn = nn.MLP.load(FILENAME)
        nn.fit(X, y, lr=lr, epochs=epochs)
        print(f"Loaded:{[nn(x).data for x in X]}")
        print(f"Actual:{y}")

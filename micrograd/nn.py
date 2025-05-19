import json
import os
import random

from micrograd.engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    # SAVE neuron data
    def _data(self):
        return {
            "b": self.b._data(),
            "il": self.nonlin,
            "w": [cw._data() for cw in self.w],
        }

    # LOAD neuron from saved data
    @staticmethod
    def data_(data):
        inst = Neuron(0, nonlin=data["il"])
        inst.w = [Value.data_(rw) for rw in data["w"]]
        inst.b = Value.data_(data["b"])

        return inst

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    # SAVE layer
    def _data(self):
        return {"ns": [n._data() for n in self.neurons]}

    # LOAD layer
    @staticmethod
    def data_(data):
        inst = Layer(0, 0)
        inst.neurons = [Neuron.data_(nr) for nr in data["ns"]]

        return inst

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

    # optional to automate training
    def fit(self, X, y, epochs=1000, lr=0.001):
        for epoch in range(epochs):
            # Forward pass
            out = [self(x) for x in X]  # Each `mp(x)` should return a `Value`

            loss = sum((ya - yp) ** 2 for yp, ya in zip(out, y)) / len(y)

            loss.backward()  # This works now because `loss` is a `Value`

            for param in self.parameters():
                param.data -= lr * param.grad  # Gradient descent update
                param.grad = 0  # Reset gradients for next iteration

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.data}")

    # SAVE mlp
    def _data(self):
        return {"ls": [ly._data() for ly in self.layers]}

    # LOAD mlp
    @staticmethod
    def data_(data):
        inst = MLP(0, [])
        inst.layers = [Layer.data_(ly) for ly in data["ls"]]
        return inst

    # Interface point for saving model
    def save(self, fp: str):
        if fp == "" or fp is None:
            raise ValueError("File path is not specified")

        with open(fp, "w", encoding="utf-8") as f:
            json.dump(self._data(), f, indent=2)
            print("SUCCESS! Model saved!")

    # Inteface point for loading model
    @staticmethod
    def load(fp: str):
        if not os.path.exists(fp):
            raise FileNotFoundError
        with open(fp, "r", encoding="utf-8") as f:
            contents = json.load(f)
            return MLP.data_(contents)

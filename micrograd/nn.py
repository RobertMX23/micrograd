import random
from micrograd.engine import Value


ACTIVATIONS = {
    'relu': lambda x: x.relu(),
    'tanh': lambda x: x.tanh(),
    'sigmoid': lambda x: x.sigmoid(),
    'linear': lambda x: x,
}


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, act = 'relu'):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.act = act
        self._actfn = self._get_act(act)

    def _get_act(self, act):
        if act not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation, only {list(ACTIVATIONS)} activations are supported")
        return ACTIVATIONS[act]
    
    def __call__(self, x):
        z = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return self._actfn(z)

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)}, act={self.act})"

class Layer(Module):

    def __init__(self, nin, nout, act = 'relu', **kwargs):
        self.neurons = [Neuron(nin, act, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts, hidden_activation = 'relu', output_activation = 'linear'):
        sz = [nin] + nouts
        acts = [hidden_activation for _ in range(len(nouts)-1)] + [output_activation]
        self.layers = [Layer(sz[i], sz[i+1], acts[i]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

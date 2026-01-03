from micrograd.engine import Value

class Optimizer:
    def __init__(self, parameters):
        self.parameters = [p for p in parameters if p.grad != 0 or True] # keep all for now

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = {p: 0 for p in self.parameters}

    def step(self):
        for p in self.parameters:
            v = self.velocities[p]
            v = self.momentum * v - self.lr * p.grad
            self.velocities[p] = v
            p.data += v

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = {p: 0 for p in self.parameters}
        self.v = {p: 0 for p in self.parameters}
        self.t = 0

    def step(self):
        self.t += 1
        for p in self.parameters:
            if p.grad == 0:
                continue
            
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * p.grad
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (p.grad ** 2)
            
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)
            
            p.data -= self.lr * m_hat / (v_hat**0.5 + self.eps)

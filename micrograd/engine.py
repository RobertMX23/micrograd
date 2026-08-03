
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op='', _local_grads=()):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._prev = _children
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self._local_grads = _local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+', (1, 1))
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*', (other.data, self.data))
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}', (other * self.data**(other-1),))
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU', (float(self.data > 0),))
        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._prev, v._local_grads):
                child.grad += local_grad * v.grad

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

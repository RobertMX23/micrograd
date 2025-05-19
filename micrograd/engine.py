class Value:
    """stores a single scalar value and its gradient"""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), (
            "only supporting int/float powers for now"
        )
        # change _op to simplify loading of the model Value.data_(data)
        out = Value(self.data**other, (self,), "**")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

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
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # SAVE: base object structure as json
    def _data(self):
        """
        Return a dict representing this Value and its computation graph.
        """
        return {
            "d": self.data,
            "o": self._op,
            "c": [child._data() for child in self._prev],
            "g": self.grad,
        }

    @staticmethod
    def data_(data):
        """
        Create a Value from a saved dictionary (as from _data).
        """
        inst = Value(data["d"], [], data["o"])
        inst.grad = data["g"]
        if not data["c"]:
            return inst
        inst._prev = []
        for cd in data["c"]:
            inst._prev.append(Value.data_(cd))

        # recreating _backward lambada based on operator and related values
        backward = lambda: None
        match data["o"]:
            case "*":

                def back():
                    inst._prev[0].grad += inst._prev[1].data * inst.grad
                    inst._prev[1].grad += inst._prev[0].data * inst.grad

                backward = back
            case "+":

                def back():
                    inst._prev[0].grad += 1.0 * inst.grad
                    inst._prev[1].grad += 1.0 * inst.grad

                backward = back
            case "**":

                def back():
                    inst._prev[0].grad += (
                        inst.data * inst._prev[0].data ** (inst.data - 1)
                    ) * inst.grad

                backward = back
            case "tanh":

                def back():
                    inst._prev[0].grad += (1 - (inst.data) ** 2) * inst.grad

                backward = back
            case "ReLU":

                def back():
                    inst._prev[0].grad += (inst.data > 0) * inst.grad

                backward = back

        inst._backward = backward
        return inst

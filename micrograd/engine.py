import numpy as np
import math

class Value:
    def __init__(self, mydata, _children=(), op='', label=''):
        self.data = mydata
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = op
        self.grad = 0
        self.label = label
    
    def __repr__(self):
        return f'Value(data={self.data})'

    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        ans = Value(self.data + other.data,(self,other), op='+')
        def _backward():
            self.grad += ans.grad
            other.grad += ans.grad
        ans._backward = _backward
        return ans
    
    def __radd__(self,other):
        return self + other

    def __mul__(self,other):
        # self = self if isinstance(self,Value) else Value(self) # its nonsense to do this to handle reverse part -> 2 * a (beacuse it is 2.__mul__(a); And you have custom __mul__ for a and not for 2)
        other = other if isinstance(other,Value) else Value(other)
        ans = Value(self.data * other.data,(self,other) , op='*')
        def _backward():
            self.grad += ans.grad*other.data
            other.grad += ans.grad*self.data
        ans._backward = _backward
        return ans
    
    def __rmul__(self, other):  # for reverse part -> other * self (Python checks for it if simple __mul__ doesn't hold)
        return self * other
    
    def __truediv__(self,other):
        return self * other**-1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        ans = Value(self.data**other, (self,), op=f'pow {other}')
        def _backward():
            self.grad += other* self.data**(other-1) * ans.grad
        ans._backward = _backward

        return ans
    
    
    def exp(self):
        x = self.data
        ans = Value(np.exp(x),(self,), op='exp')
        def _backward():
            self.grad += ans.grad * ans.data 

        ans._backward = _backward

        return ans

    def tanh(self):
        x = self.data
        t = (np.exp(2*x) - 1)/(np.exp(2*x) + 1)
        ans = Value(t,(self,),'tanh')
        def _backward():
            self.grad += ans.grad*(1-t**2)
            # self._backward()
        ans._backward = _backward

        return ans
    
    def relu(self):
        r = self.data
        if (r<0):
            r = 0
        ans = Value(r, (self,), op='ReLU')

        def _backward():
            self.grad += ans.grad * (self.data>0)

        ans._backward = _backward

        return ans
    
    # def softmax(self):
        
    # def log(self):
    #     assert self.data > 0, "Logarithm is undefined for non-positive values"
    #     ans = Value(np.log(self.data), (self,), op="log")

    #     def _backward():
    #         self.grad += (ans.grad / self.data)  # d/dx (ln x) = 1/x

    #     ans._backward = _backward
    #     return ans

    def log(self, epsilon=1e-7):
        ans = Value(math.log(self.data + epsilon), (self,), 'log')

        def _backward():
            self.grad += (1/(self.data + epsilon)) * ans.grad
            
        ans._backward = _backward

        return ans

    
    def __neg__(self): # -self
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other): # other - self (eg: 1 - a)
        # print("I came here")
        return (-self) + other
        # return other - self
    
    # def __lt__(self, other):
    #     other = other if isinstance(other, Value) else Value(other)
    #     return self.data < other.data

    def __le__(self, other):
        # other = other if isinstance(other, Value) else Value(other)
        return self.data <= other.data

    # def __eq__(self, other):
    #     other = other if isinstance(other, Value) else Value(other)
    #     return self.data == other.data

    # def __ne__(self, other):
    #     other = other if isinstance(other, Value) else Value(other)
    #     return self.data != other.data

    def __ge__(self, other):
        # other = other if isinstance(other, Value) else Value(other)
        return self.data >= other.data

    def __gt__(self, other):
        # other = other if isinstance(other, Value) else Value(other)
        return self.data > other.data
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(i):
            if i not in visited:
                # visited.insert(i)
                visited.add(i)
                for prev in i._prev:
                    build_topo(prev)
                topo.append(i)

        build_topo(self)

        self.grad = 1
        for i in reversed(topo):
            i._backward()
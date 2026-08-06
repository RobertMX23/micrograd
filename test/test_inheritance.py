import micrograd
import math

def test_new_method():
    """
    Creates a new class that inherits with 
        same parameters
        new method
    """
    class Value(micrograd.Value):

        def tanh(self):
            x = self.data
            temp = math.exp(2*x)
            t = (temp - 1) / (temp + 1)
            out = self.__class__(data=t, _children = (self,), label = "tanh")
            return out

    n = Value(data = 0.6, label = "neuron")
    is_successful = True
    try:
        n.tanh()
    except AttributeError as error:
        is_successful = False
        raise(error)

    assert is_successful
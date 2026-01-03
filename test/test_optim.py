import unittest
import torch
from micrograd.engine import Value
from micrograd.optim import SGD, Adam

class TestOptim(unittest.TestCase):

    def test_sgd(self):
        # Simple quadratic optimization: y = (x - 3)^2
        # Minimum at x = 3
        
        # micrograd
        x = Value(0.0)
        optimizer = SGD([x], lr=0.1)
        
        for _ in range(100):
            optimizer.zero_grad()
            loss = (x - 3)**2
            loss.backward()
            optimizer.step()
            
        self.assertAlmostEqual(x.data, 3.0, delta=0.1)

    def test_adam(self):
        # Simple quadratic optimization: y = (x - 3)^2
        # Minimum at x = 3
        
        # micrograd
        x = Value(0.0)
        optimizer = Adam([x], lr=0.1)
        
        for _ in range(300):
            optimizer.zero_grad()
            loss = (x - 3)**2
            loss.backward()
            optimizer.step()
            
        self.assertAlmostEqual(x.data, 3.0, delta=0.01)

if __name__ == '__main__':
    unittest.main()

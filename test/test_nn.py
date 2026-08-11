import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP


class TestNN(unittest.TestCase):

    def test_neuron_forward_and_params(self):
        n = Neuron(3, nonlin='relu')
        x = [Value(1.0), Value(-2.0), Value(0.5)]
        out = n(x)
        self.assertIsInstance(out, Value)
        self.assertEqual(len(n.parameters()), 4)  # 3 weights + 1 bias

    def test_neuron_activations(self):
        # Linear neuron
        n_lin = Neuron(2, nonlin=False)
        n_lin.w = [Value(2.0), Value(-1.0)]
        n_lin.b = Value(0.5)
        res_lin = n_lin([Value(1.0), Value(3.0)])
        self.assertAlmostEqual(res_lin.data, 2.0 * 1.0 + (-1.0) * 3.0 + 0.5)

        # Tanh neuron
        n_tanh = Neuron(1, nonlin='tanh')
        n_tanh.w = [Value(0.0)]
        n_tanh.b = Value(0.0)
        self.assertAlmostEqual(n_tanh([Value(5.0)]).data, 0.0)

    def test_layer(self):
        layer = Layer(3, 2)
        x = [Value(1.0), Value(2.0), Value(-1.0)]
        out = layer(x)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(layer.parameters()), 2 * (3 + 1))

    def test_mlp_training_step(self):
        mlp = MLP(2, [4, 1])
        x = [Value(2.0), Value(3.0)]
        y = mlp(x)
        self.assertIsInstance(y, Value)

        # Calculate loss and run backprop
        loss = (y - 1.0) ** 2
        loss.backward()

        # Check gradients are computed
        params = mlp.parameters()
        self.assertTrue(any(p.grad != 0 for p in params))

        # Test zero_grad
        mlp.zero_grad()
        self.assertTrue(all(p.grad == 0 for p in params))

    def test_repr(self):
        n = Neuron(2, nonlin='tanh')
        self.assertIn("tanhNeuron", str(n))
        mlp = MLP(2, [3, 1])
        self.assertIn("MLP", str(mlp))


if __name__ == '__main__':
    unittest.main()

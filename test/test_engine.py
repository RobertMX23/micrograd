import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from micrograd.engine import Value

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestEngine(unittest.TestCase):

    def test_sanity_check(self):
        x = Value(-4.0)
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        y.backward()

        if HAS_TORCH:
            x_pt = torch.Tensor([-4.0]).double()
            x_pt.requires_grad = True
            z_pt = 2 * x_pt + 2 + x_pt
            q_pt = z_pt.relu() + z_pt * x_pt
            h_pt = (z_pt * z_pt).relu()
            y_pt = h_pt + q_pt + q_pt * x_pt
            y_pt.backward()

            self.assertEqual(y.data, y_pt.data.item())
            self.assertEqual(x.grad, x_pt.grad.item())

    def test_more_ops(self):
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
        g.backward()

        tol = 1e-6
        if HAS_TORCH:
            a_pt = torch.Tensor([-4.0]).double()
            b_pt = torch.Tensor([2.0]).double()
            a_pt.requires_grad = True
            b_pt.requires_grad = True
            c_pt = a_pt + b_pt
            d_pt = a_pt * b_pt + b_pt**3
            c_pt = c_pt + c_pt + 1
            c_pt = c_pt + 1 + c_pt + (-a_pt)
            d_pt = d_pt + d_pt * 2 + (b_pt + a_pt).relu()
            d_pt = d_pt + 3 * d_pt + (b_pt - a_pt).relu()
            e_pt = c_pt - d_pt
            f_pt = e_pt**2
            g_pt = f_pt / 2.0
            g_pt = g_pt + 10.0 / f_pt
            g_pt.backward()

            self.assertTrue(abs(g.data - g_pt.data.item()) < tol)
            self.assertTrue(abs(a.grad - a_pt.grad.item()) < tol)
            self.assertTrue(abs(b.grad - b_pt.grad.item()) < tol)

    def test_tanh(self):
        x = Value(0.5)
        y = x.tanh()
        y.backward()

        expected_y = math.tanh(0.5)
        expected_grad = 1.0 - expected_y**2
        self.assertAlmostEqual(y.data, expected_y, places=6)
        self.assertAlmostEqual(x.grad, expected_grad, places=6)

        if HAS_TORCH:
            x_pt = torch.Tensor([0.5]).double()
            x_pt.requires_grad = True
            y_pt = torch.tanh(x_pt)
            y_pt.backward()
            self.assertAlmostEqual(y.data, y_pt.data.item(), places=6)
            self.assertAlmostEqual(x.grad, x_pt.grad.item(), places=6)

    def test_exp_and_log(self):
        x = Value(1.5)
        y = x.exp()
        z = y.log()
        z.backward()

        self.assertAlmostEqual(y.data, math.exp(1.5), places=6)
        self.assertAlmostEqual(z.data, 1.5, places=6)
        self.assertAlmostEqual(x.grad, 1.0, places=6)

        if HAS_TORCH:
            x_pt = torch.Tensor([1.5]).double()
            x_pt.requires_grad = True
            y_pt = torch.exp(x_pt)
            z_pt = torch.log(y_pt)
            z_pt.backward()
            self.assertAlmostEqual(z.data, z_pt.data.item(), places=6)
            self.assertAlmostEqual(x.grad, x_pt.grad.item(), places=6)

    def test_sigmoid(self):
        x = Value(0.8)
        y = x.sigmoid()
        y.backward()

        expected_y = 1.0 / (1.0 + math.exp(-0.8))
        expected_grad = expected_y * (1.0 - expected_y)
        self.assertAlmostEqual(y.data, expected_y, places=6)
        self.assertAlmostEqual(x.grad, expected_grad, places=6)

        if HAS_TORCH:
            x_pt = torch.Tensor([0.8]).double()
            x_pt.requires_grad = True
            y_pt = torch.sigmoid(x_pt)
            y_pt.backward()
            self.assertAlmostEqual(y.data, y_pt.data.item(), places=6)
            self.assertAlmostEqual(x.grad, x_pt.grad.item(), places=6)


if __name__ == '__main__':
    unittest.main()

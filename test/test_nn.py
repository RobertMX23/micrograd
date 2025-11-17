"""Test implementation of the MLP object"""

from pytest import approx
from micrograd.nn import MLP
from micrograd.engine import Value


def test_simple_mlp():
    """Test simple nn object returns the expected values"""    
    model = MLP(1, [1,1])
    model.layers[0].neurons[0].w = [Value(1.3)]
    model.layers[0].neurons[0].b = Value(4)
    model.layers[1].neurons[0].w = [Value(2.4)]
    model.layers[1].neurons[0].b = Value(2)

    y_hat = model([2.0])
    assert approx(y_hat.data) == 17.84

    loss = (8.84 - y_hat)**2
    assert approx(loss.data) == 81
    loss.backward()

    expected_grad = {
        "1.3": 86.4,
        "4": 43.2,
        "2.4": 118.8,
        "2": 18,
    }

    for p in model.parameters():
        assert expected_grad.get(str(p.data)) == approx(p.grad)

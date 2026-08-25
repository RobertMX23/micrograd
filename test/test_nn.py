from micrograd.engine import Value
from micrograd.nn import Layer, MLP


def test_mlp_hidden_width_one_forward():
    """A 1-neuron hidden layer used to crash: Layer unwraps to a Value."""
    model = MLP(2, [1, 1])
    x = [Value(0.5), Value(-1.5)]
    y = model(x)
    assert isinstance(y, Value)


def test_mlp_hidden_width_one_then_wider():
    model = MLP(2, [1, 4])
    y = model([Value(1.0), Value(-2.0)])
    assert isinstance(y, list) and len(y) == 4
    assert all(isinstance(v, Value) for v in y)


def test_mlp_single_output_still_scalar():
    model = MLP(2, [3, 1])
    y = model([Value(0.25), Value(0.75)])
    assert isinstance(y, Value)


def test_layer_single_neuron_still_scalar():
    layer = Layer(2, 1, nonlin=False)
    y = layer([Value(1.0), Value(2.0)])
    assert isinstance(y, Value)


def test_mlp_hidden_width_one_backward():
    model = MLP(2, [1, 1])
    y = model([Value(0.3), Value(-0.7)])
    y.backward()
    params = model.parameters()
    assert len(params) == 2 * 1 + 1 + 1 * 1 + 1  # w/b hidden + w/b output
    assert any(p.grad != 0 for p in params)

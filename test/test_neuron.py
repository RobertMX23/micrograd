from micrograd.nn import Neuron
import torch
import torch.nn as nn


def test_neuron_build():

    X = [2.0, 3.0, -1.0]
    n = Neuron(3, 'relu')
    mgoutputs = n(X)
    print(mgoutputs)


    XP = torch.tensor(X)
    nt = nn.Sequential(
        nn.Linear(in_features=3, out_features=1),
        nn.ReLU()
    )

    ptoutputs = nt(XP)
    print(ptoutputs)
from micrograd.nn import Neuron, Layer, MLP
import torch
import torch.nn as nn

def main():

    X = [2.0, 3.0, 1.0]
    n = Neuron(3, 'sigmoid')
    print(n)
    mgoutputs = n(X)
    print(mgoutputs)

    mlp = MLP(3, [4, 4, 1], hidden_activation='relu', output_activation='sigmoid')
    print(mlp)
    mlpout = mlp(X)
    print(mlpout)

    XP = torch.tensor(X)
    nt = nn.Sequential(
        nn.Linear(in_features=3, out_features=1),
        nn.Sigmoid()
    )

    ptoutputs = nt(XP)
    print(ptoutputs)





if __name__ == '__main__':
    main()

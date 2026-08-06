from engine import Value
import numpy as np
from sklearn.metrics import accuracy_score

class Neuron:

    def __init__(self, nin:int, act_fn:str = "relu"):
        # limit = np.sqrt(6 / nin)
        # self.w = [Value(np.random.uniform(-limit, limit)) for _ in range(nin)]
        # self.w = [Value(np.random.normal(-1,1)) for i in range(nin)]
        self.w = [Value(np.random.uniform(-1,1)) for i in range(nin)]
        self.b = Value(0)
        self.act_fn = act_fn
    
    def __call__(self,x):
        # out = sum((wi*xi for wi, xi in zip(self.w, x))) + self.b
        out = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) # efficient than above line as sum takes an option 2nd argument, that is by default  = 0

        if (self.act_fn=="relu"):
            activ = out.relu()
            return activ
        
        elif self.act_fn=="tanh":
            activ = out.tanh()
            return activ
        
        else:
            activ = out
            return activ
    
    def parameters(self):
      return self.w + [self.b]
    

class Layer:

    def __init__(self, nin:int, nout:int, activation_fn:str):  # nout for no. of neurons = output.size
        self.neurons = [Neuron(nin, activation_fn) for i in range(nout)]
        self.activ_fn = activation_fn

    def __call__(self, x):
        if (self.activ_fn=="softmax"):
            # print("I was here")
            out = [n(x) for n in self.neurons]
            # max_value = max(out)

            max_value = Value(max([v.data for v in out]))

            norm_out = [(it - max_value).exp() for it in out]
            denom = sum(norm_out)
            # for val in range(1,len(norm_out)):
            #     denom += norm_out[val]
            ans =  [(it/denom) for it in norm_out]
            return ans[0] if len(ans)==1 else ans
        
        else:
            ans2 = [n(x) for n in self.neurons]
            return ans2[0] if len(ans2)==1 else ans2

        # return out[0] if (len(out)==1) else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    

class MLP:
    
    def __init__(self, nin, nouts:list, activations:list):
        netsize = [nin] + nouts
        self.Layers = [Layer(netsize[i],netsize[i+1],activations[i]) for i in range(len(nouts))]
        self.nin = nin
        # self.ytrue = y


    def __call__(self, x):
        out = x
        for Li in self.Layers:
            out = Li(out)
        return out
    
    def MSE(self,y, y_pred):
        Loss = sum([(yp-yt)**2 for yp,yt in zip(y_pred, y)])
        return Loss

    def cross_entropy_loss(self,y, y_pred):
        n = len(y)
        cumloss = 0
        y_for_accuracy = []
        for ind, yind in enumerate(y):
            cumloss -= y_pred[ind][yind].log()
            y_for_accuracy.append(np.argmax(y_pred[ind]).item())
        # loss = -sum([y_pred[ind][yind].log() for ind, yind in enumerate(y)]) /n
        loss = cumloss / n
        acc = accuracy_score(y,y_for_accuracy)
        
        return loss, acc
        
    def parameters(self):
        return [p for layer in self.Layers for p in layer.parameters()]

    def train(self, X, y, epochs, batch_size=30):
        num_samples = len(X)
        num_batches = num_samples // batch_size  # Number of batches per epoch

        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for i in range(num_batches):
                # indices
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # forward-pass
                y_pred_batch = np.array([self(xi) for xi in X_batch])
                Loss, acc = self.cross_entropy_loss(y_batch, y_pred_batch)

                # zero-grad
                for param in self.parameters():
                    param.grad = 0

                # back-pass
                Loss.backward()

                # upadte
                learning_rate = 1.0 - 0.9 * (epoch * num_batches + i) / (epochs * num_batches)
                for param in self.parameters():
                    param.data -= learning_rate * param.grad

                print(f"Iter : {i} | Loss : {Loss.data} | Accuracy : {acc*100}")

            # Print loss/accuracy for monitoring
            print("-"*15)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {Loss}, Accuracy: {acc}")
            print("-"*15)



        
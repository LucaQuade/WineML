import numpy as np


class Module:
    
    def forward(self, *args, **kwargs):
        pass
        
    def backward(self, *args, **kwargs):
        pass

    
class Network(Module):
    
    def __init__(self, layers=None):
        self.layers = layers
    
    def forward(self, x, target):
    
        #feed output of a layer as input to the next layer
        for l in self.layers[:-1]:
            x = l.forward(x)
        prediction, loss = self.layers[-1].forward(x, target)
        return prediction, loss
        
    def backward(self, prediction, target):
    
        #pass derivatives backwards through the layers
        gY = self.layers[-1].backward(prediction, target)
        for l in self.layers[-2::-1]:
            gY = l.backward(gY)
        return gY
    
    def add_layer(self, layer):
        self.layers.append(layer)

    
class LinearLayer(Module):
    
    def __init__(self, W, b):
        self.W = W
        self.b = b
    
    def forward(self, X):
        self.X = X
        return (self.W @ X) + self.b
        
    def backward(self, gY):
        pass
        #TO-DO

    
class Sigmoid(Module):
    
    def forward(self, x):
        return np.exp(x) / (1 + np.exp(x))
        
    def backward(self,x):
        
        return self.forward(x) * (1 - self.forward(x))

    
class ReLU(Module):
    
    def forward(self, x):
    
        #set negative values to 0
        return np.maximum(x, 0)
        
    def backward(self, x):
    
        #derivative of ReLU is technically undefined for x=0. We assume x'=0 for x=0
        x = [1 if x_ > 0 else 0 for x_ in x]
        return x

    
class Loss(Module):
    
    def forward(self, prediction, target):
        pass
        
    def backward(self, prediction, target):
        pass


class MSE(Loss):
    
    def forward(self, prediction, target):
        return prediction, np.mean(((prediction - target) ** 2))
        
    def backward(self, prediction, target):
        return 2 * (prediction - target) / prediction.shape[0]


class CrossEntropyLoss(Loss):
    
    def forward(self, prediction, target):
        
        #softmax
        cross_entropy = np.exp(prediction) / np.exp(prediction).sum()
        
        return - np.log(cross_entropy[target])
        
    def backward(self, prediction, target):
        pass
        #To-Do
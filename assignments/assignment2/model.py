import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        self.layersList = [FullyConnectedLayer(n_input, hidden_layer_size), ReLULayer(),  FullyConnectedLayer(hidden_layer_size, n_output) ]
        self.reg = reg

    def compute_loss_and_gradients(self, X, y):
        for every in self.params():
            self.params()[every].grad = 0
        
        copyX = X.copy()
        
        for every in self.layersList:
            copyX = every.forward(copyX)
        
        loss, d_out = softmax_with_cross_entropy(copyX, y)
        
        for every in reversed(self.layersList):
            d_out = every.backward(d_out)
        
        for every in self.params():
            lossPart, gradPart = l2_regularization(self.params()[every].value, self.reg)
            self.params()[every].grad += gradPart
            loss += lossPart
            
        return loss

    def predict(self, X):
        pred = np.zeros(X.shape[0], np.int)
        
        for every in self.params():
            self.params()[every].grad = 0
        
        copyX = X.copy()
        
        for every in self.layersList:
            copyX = every.forward(copyX)
        
        pred = np.argmax(copyX, axis = 1)
        
        return pred
        
    def params(self):
        result = {
            'inputLayerW':self.layersList[0].params()['W'],
            'inputLayerB':self.layersList[0].params()['B'],
            'outputLayerW':self.layersList[2].params()['W'],
            'outputLayerB':self.layersList[2].params()['B']
        }


        return result

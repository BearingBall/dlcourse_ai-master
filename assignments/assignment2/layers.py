import numpy as np

def softmax(predictions):
    if (predictions.ndim == 1):
        predictionsTmp = np.copy(predictions)
        predictionsTmp -= np.max(predictions) 
        divider = 0;
        for i in range(len(predictions)):
            divider += np.exp(predictionsTmp[i])
        probs = np.exp(predictionsTmp)/divider
        return probs
    else: 
        predictionsTmp = np.copy(predictions)
        predictionsTmp = (predictionsTmp.T - np.max(predictions,axis = 1)).T 
        exp_pred = np.exp(predictionsTmp) 
        exp_sum= np.sum(exp_pred,axis=1) 
        return (exp_pred.T / exp_sum).T


def cross_entropy_loss(probs, target_index):
    if (probs.ndim == 1):
        loss = -np.log(probs[target_index])
    else:
        batch_size = probs.shape[0]
        loss_arr = -np.log(probs[range(batch_size), target_index])
        loss = np.sum(loss_arr) / batch_size
        
    return loss


def l2_regularization(W, reg_strength):
    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * reg_strength * W

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs
    if (probs.ndim == 1):
        dprediction[target_index] -= 1
    else:
        batch_size = predictions.shape[0]
        dprediction[range(batch_size), target_index] -= 1
        dprediction /= batch_size
    return loss, dprediction


class Param:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None
        pass

    def forward(self, X):
        self.X = X
        res = X.copy()
        for x in np.nditer(res, op_flags=['readwrite']) :
            x[...] = x if x>=0 else 0
        return res

    def backward(self, d_out):
        res = self.X.copy()
        for x in np.nditer(res, op_flags=['readwrite']):
            x[...] = 1 if x>=0 else 0
        return res*d_out

    def params(self):
        return {}

class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X,self.W.value)+self.B.value

    def backward(self, d_out):
        d_input = np.dot(d_out, self.W.value.transpose())
        self.W.grad += np.dot(self.X.transpose(), d_out)       
        self.B.grad += np.dot(np.ones((1, d_out.shape[0])), d_out)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
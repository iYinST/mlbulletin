import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


def ReLu(x):
    return np.where(x > 0, x, 0.0001 * x)


def ReLu_deriv(x):
    return np.where(x > 0, 1.0, 0.0001)


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def Mean_Squared_Error(yhat,y):
    return (yhat - y) ** 2

def Mean_Squared_Error_diriv(yhat,y):
    return 2 * (yhat - y)

def Mean_Absolute_Error(yhat,y):
    return abs(yhat - y)

def Mean_Absolute_Error_diriv(yhat,y):
    return np.where(yhat > y, 1.0, -1.0)

class NeuralNetwork:
    def __init__(self, layers, activation='tanh', normalization=False):
        if activation == 'sig':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        elif activation == 'ReLu':
            self.activation = ReLu
            self.activation_deriv = ReLu_deriv
        self.normalization = normalization
        self.weights = []
        self.layers = layers
        self.inited = False

    def init(self, type='normal'):
        self.inited = True
        if type == 'zeros':
            for i in range(1, len(self.layers) - 1):
                self.weights.append(np.zeros((self.layers[i - 1] + 1, self.layers[i] + 1)))
                self.weights.append(np.zeros((self.layers[i] + 1, self.layers[i + 1])))
        elif type == 'normal':
            for i in range(1, len(self.layers) - 1):
                self.weights.append((2 * np.random.random((self.layers[i - 1] + 1, self.layers[i] + 1)) - 1) * 0.25)
                self.weights.append((2 * np.random.random((self.layers[i] + 1, self.layers[i + 1])) - 1) * 0.25)
        elif type == 'other':
            for i in range(1, len(self.layers) - 1):
                self.weights.append((2 * np.random.random((self.layers[i - 1] + 1, self.layers[i] + 1)) + 0) * 0.25)
                self.weights.append((2 * np.random.random((self.layers[i] + 1, self.layers[i + 1])) + 0) * 0.25)

    def fit(self, X, Y, lr=0.01, epochs=10000, normalization=False,loss = 'mse'):
        if not self.inited:
            self.init()
        self.normalization = normalization
        # Z - score
        if normalization:
            self.x_mu = np.mean(X, axis=0)
            self.x_std = np.std(X, axis=0)
            self.y_mu = np.mean(Y)
            self.y_std = np.std(Y)
            Y = (Y - self.y_mu)/ self.y_std
            X = (X - self.x_mu)/ self.x_std
        if loss == 'mse':
            self.loss_fun = Mean_Squared_Error
            self.loss_fun_deriv = Mean_Squared_Error_diriv
        elif loss == 'mae':
            self.loss_fun = Mean_Absolute_Error
            self.loss_fun_deriv = Mean_Absolute_Error_diriv
        X = np.atleast_2d(X)
        tmp = np.ones([X.shape[0], X.shape[1] + 1])
        tmp[:, 0:-1] = X
        X = tmp
        Y = np.array(Y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = Y[i] - a[-1]
            # mse
            error = self.loss_fun(Y[i],a[-1])
            deltas = [error * self.activation_deriv(a[-1]) * self.loss_fun_deriv(Y[i],a[-1])]

            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            deltas.reverse()

            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += lr * layer.T.dot(delta)

    def predict(self, x):
        if self.normalization:
            x = (x - self.x_mu)/ self.x_std
        x = np.array(x)
        tmp = np.ones(x.shape[0] + 1)
        tmp[0:-1] = x
        a = tmp

        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        if self.normalization:
            return a * self.y_std + self.y_mu
        return a


if __name__ == "__main__":
    nn = NeuralNetwork([2, 2, 2], 'tanh')
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    nn.fit(X, Y)
    for i in X:
        print(i, nn.predict(i))

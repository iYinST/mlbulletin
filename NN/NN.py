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
    # return x
    return np.where(x > 0, x, 0.0001*x)
    return np.where(x > 0, x, 0.0)


def ReLu_deriv(x):
    # return 1
    return np.where(x > 0, 1.0, 0.0001)
    return np.where(x > 0, 1.0, 0.0)


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


class NeuralNetwork:
    def __init__(self, layers, activation='tanh',normalization = False):
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

        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) + 0) * 0.25)
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) + 0) * 0.25)

    def fit(self, X, Y, lr=0.01, epochs=10000):
        X = np.atleast_2d(X)
        tmp = np.ones([X.shape[0], X.shape[1] + 1])
        tmp[:, 0:-1] = X
        X = tmp
        Y = np.array(Y)

        t = X.shape[1]
        # if self.normalization:


        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = Y[i] - a[-1]

            deltas = [error * self.activation_deriv(a[-1])]

            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            deltas.reverse()

            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += lr * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        tmp = np.ones(x.shape[0] + 1)
        tmp[0:-1] = x
        a = tmp

        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


if __name__ == "__main__":
    nn = NeuralNetwork([2, 2, 2], 'tanh')
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    nn.fit(X, Y)
    for i in X:
        print(i, nn.predict(i))

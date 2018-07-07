from NN import  NN
import numpy as np
from hosing import hosing

if __name__ == "__main__":
    X,Y = hosing.load()

    test_X = X[-50:]
    test_Y = Y[-50:]
    train_X = X[:-50]
    train_Y = Y[:-50]

    nn = NN.NeuralNetwork([13,10,1],'ReLu')
    nn.fit(train_X,train_Y,epochs=500000,lr= 0.001)
    right = 0
    all = 0
    for i in zip(test_X,test_Y):
        result = nn.predict(i[0])
        print(result[0],i[1])
        all+= 1

    print("precision",right/all)

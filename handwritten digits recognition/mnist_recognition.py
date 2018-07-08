from NN import  NN
from dataset.mnist import mnist
import numpy as np
if __name__ == "__main__":
    # load the data
    train_images, train_labels = mnist.load_train()
    test_images, test_labels = mnist.load_test()

    # crate nn model
    nn = NN.NeuralNetwork([784,100,10],'sig')
    # init
    nn.init('normal')
    # fit
    nn.fit(train_images,train_labels,epochs=5000,lr=0.1,loss='mae',batch=1)

    # predict
    right = 0
    all = 0
    for i in zip(test_images,test_labels):
        result = nn.predict(i[0])
        if np.argmax(result) == np.argmax(i[1]):
            right += 1
        all+= 1

    print("precision",right/all)

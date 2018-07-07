from NN import  NN
from mnist import mnist
import numpy as np
if __name__ == "__main__":
    train_images, train_labels = mnist.load_train()
    test_images, test_labels = mnist.load_test()

    nn = NN.NeuralNetwork([784,300,10],'sig')
    nn.fit(train_images,train_labels,epochs=30000)
    right = 0
    all = 0
    for i in zip(test_images,test_labels):
        result = nn.predict(i[0])
        if np.argmax(result) == np.argmax(i[1]):
            right += 1
        all+= 1

    print("precision",right/all)

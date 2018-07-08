from NN import  NN
from dataset.hosing import hosing

if __name__ == "__main__":
    # load the train data
    X,Y = hosing.load()

    # data partition
    test_X = X[-50:]
    test_Y = Y[-50:]
    train_X = X[:-50]
    train_Y = Y[:-50]

    # create nn model
    nn = NN.NeuralNetwork([13,10,1],'ReLu')
    # init
    nn.init('other')
    # fit
    nn.fit(train_X,train_Y,epochs=10000,lr= 0.0001,normalization=True,batch = 16,loss='mse')
    # predict
    right = 0
    all = 0
    mae = 0
    for i in zip(test_X,test_Y):
        result = nn.predict(i[0])
        print(result[0],i[1])
        mae += abs(result[0]-i[1])
        all+= 1

    print("mae",mae/all)


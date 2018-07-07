import numpy as np


def load():
    result = []
    with open('./Housing', 'r') as f:
        content = f.readline()
        while content:
            result.append(content.split())
            content = f.readline()
    result = np.array(result)
    result = result.astype('float64')
    return np.array(result[:,:-1]),np.array(result[:,-1])


if __name__ == "__main__":
    print(load())

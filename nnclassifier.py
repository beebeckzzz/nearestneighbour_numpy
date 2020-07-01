import numpy as np

class nnclassifier:

    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.m = np.array(X).shape[0]


    def predict(self):

        y_pred = np.zeros(self.m)
        for i in range(self.m):
            temp = np.delete(self.X, i, axis = 0)
            distance = np.sum(np.abs(temp - self.X[i, :]), axis = 1)
            min_index = np.argmin(distance)
            y_pred[i] = self.y[min_index]
        acc = np.mean(y_pred == self.y)
        cache = {"output" : y_pred, "accuracy" : acc}

A = np.random.randint(0, 256,  [10000, 784])
b = np.random.randint(0, 2, 10000)
c = nnclassifier(A,b)
print(c.predict())

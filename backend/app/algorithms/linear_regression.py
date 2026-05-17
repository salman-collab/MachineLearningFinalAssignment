import random

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def initialize_weights(self, n_features):
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(n_features)]
        self.bias = 0

    def predict_single(self, x):
        y_pred = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return y_pred

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def compute_loss(self, y_true, y_pred):
        n = len(y_true)
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])

        self.initialize_weights(n_features)

        for epoch in range(self.epochs):
            y_pred = self.predict(X)

            # gradients
            dw = [0] * n_features
            db = 0

            for i in range(n_samples):
                error = y_pred[i] - y[i]

                for j in range(n_features):
                    dw[j] += error * X[i][j]

                db += error

            # average gradients
            dw = [d / n_samples for d in dw]
            db = db / n_samples

            # update weights
            self.weights = [w - self.lr * d for w, d in zip(self.weights, dw)]
            self.bias -= self.lr * db

            # print loss occasionally
            if epoch % 100 == 0:
                loss = self.compute_loss(y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss}")

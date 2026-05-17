import math
import random


class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def initialize_weights(self, n_features):
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(n_features)]
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def predict_proba_single(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self.sigmoid(z)

    def predict_proba(self, X):
        return [self.predict_proba_single(x) for x in X]

    def predict(self, X):
        probs = self.predict_proba(X)
        return [1 if p >= 0.5 else 0 for p in probs]


    def compute_loss(self, y_true, y_pred):
        n = len(y_true)
        loss = 0

        for yt, yp in zip(y_true, y_pred):
            yp = max(min(yp, 1 - 1e-15), 1e-15)  # avoid log(0)
            loss += yt * math.log(yp) + (1 - yt) * math.log(1 - yp)

        return -loss / n

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])

        self.initialize_weights(n_features)

        for epoch in range(self.epochs):
            y_pred = self.predict_proba(X)

            dw = [0] * n_features
            db = 0

            for i in range(n_samples):
                error = y_pred[i] - y[i]

                for j in range(n_features):
                    dw[j] += error * X[i][j]

                db += error

            dw = [d / n_samples for d in dw]
            db = db / n_samples

            self.weights = [w - self.lr * d for w, d in zip(self.weights, dw)]
            self.bias -= self.lr * db

            if epoch % 100 == 0:
                loss = self.compute_loss(y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss}")
# Logistic regression class
import numpy as np
from sklearn.tree import DecisionTreeRegressor




# Logistic Regression from Scratch


class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=5000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        n_samples, n_features = X.shape

        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_prob(self, X):
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probs = self.predict_prob(X)
        return (probs >= threshold).astype(int).ravel()






# Random Forest from Scratch (with Decision Stumps)


class SimpleDecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None

    def fit(self, X, y):
        X = np.array(X)
        self.feature_index = np.random.randint(0, X.shape[1])
        self.threshold = np.mean(X[:, self.feature_index])

    def predict(self, X):
        X = np.array(X)
        return (X[:, self.feature_index] > self.threshold).astype(int)


class RandomForestScratch:
    def __init__(self, n_trees=10):
        self.n_trees = n_trees
        self.trees = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.trees = []

        for _ in range(self.n_trees):
            stump = SimpleDecisionStump()
            stump.fit(X, y)
            self.trees.append(stump)

    def predict(self, X):
        X = np.array(X)

        tree_predictions = np.array([
            tree.predict(X) for tree in self.trees
        ])

        return np.round(np.mean(tree_predictions, axis=0)).astype(int)





# XGBoost (Gradient Boosting) from Scratch

class XGBoostScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        # Initialize predictions with 0.5 probability
        y_pred = np.full(y.shape, 0.0)

        self.trees = []

        for _ in range(self.n_estimators):
            # Gradient of log loss
            prob = self.sigmoid(y_pred)
            grad = prob - y

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, grad)

            update = tree.predict(X).reshape(-1, 1)

            y_pred -= self.learning_rate * update
            self.trees.append(tree)

    def predict_prob(self, X):
        X = np.array(X)

        y_pred = np.zeros((X.shape[0], 1))

        for tree in self.trees:
            update = tree.predict(X).reshape(-1, 1)
            y_pred -= self.learning_rate * update

        return self.sigmoid(y_pred)

    def predict(self, X, threshold=0.5):
        probs = self.predict_prob(X)
        return (probs >= threshold).astype(int).ravel()

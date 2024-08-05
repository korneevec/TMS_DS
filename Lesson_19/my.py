import numpy as np


def split_dataset(X, y, feature_index, threshold):
    left = X[:, feature_index] <= threshold
    right = X[:, feature_index] > threshold
    return (X[left], y[left]), (X[right], y[right])


def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1.0 - np.sum(probabilities**2)


def best_split(X, y):
    m, n = X.shape
    best_gini = 1
    best_index = None
    best_threshold = None

    for feature_index in range(n):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            (left_X, left_y), (right_X, right_y) = split_dataset(
                X, y, feature_index, threshold
            )
            if len(left_y) == 0 or len(right_y) == 0:
                continue

            p_left = len(left_y) / len(y)
            p_right = len(right_y) / len(y)
            gini_split = p_left * gini(left_y) + p_right * gini(right_y)

            if gini_split < best_gini:
                best_gini = gini_split
                best_index = feature_index
                best_threshold = threshold

    return best_index, best_threshold


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict(inputs) for inputs in X])

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if (
            depth >= self.max_depth
            or num_labels == 1
            or num_samples < self.min_samples_split
        ):
            return Node(value=self._most_common_label(y))

        feature_index, threshold = best_split(X, y)
        if feature_index is None:
            return Node(value=self._most_common_label(y))

        (left_X, left_y), (right_X, right_y) = split_dataset(
            X, y, feature_index, threshold
        )
        left = self._grow_tree(left_X, left_y, depth + 1)
        right = self._grow_tree(right_X, right_y, depth + 1)
        return Node(feature_index, threshold, left, right)

    def _predict(self, inputs):
        node = self.tree
        while node.value is None:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _most_common_label(self, y):
        unique, counts = np.unique(y, return_counts=True)
        most_common = unique[np.argmax(counts)]
        return most_common

    def print_tree(self):
        self._print_tree(self.tree)

    def _print_tree(self, node, depth=0):
        if node.value is not None:
            print("  " * depth + f"Leaf: {node.value}")
        else:
            print("  " * depth + f"Feature {node.feature_index} <= {node.threshold}")
            print("  " * (depth + 1) + "Left:")
            self._print_tree(node.left, depth + 1)
            print("  " * (depth + 1) + "Right:")
            self._print_tree(node.right, depth + 1)


class Node:
    def __init__(
        self, feature_index=None, threshold=None, left=None, right=None, value=None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class Node:
    def __init__(
        self, feature_index=None, threshold=None, left=None, right=None, value=None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class GradientBoosting:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.y_pred = np.zeros(y.shape)

        residuals = y - sigmoid(self.y_pred)

        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            tree_predictions = tree.predict(X)
            tree_predictions = np.clip(tree_predictions, 0, 1)

            self.y_pred += self.learning_rate * tree_predictions
            residuals = y - sigmoid(self.y_pred)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            tree_predictions = tree.predict(X)
            tree_predictions = np.clip(tree_predictions, 0, 1)
            predictions += self.learning_rate * tree_predictions
        return (sigmoid(predictions) > 0.5).astype(int)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # clf = DecisionTree(max_depth=3, min_samples_split=2)
    # clf.fit(X_train, y_train)

    # y_pred = clf.predict(X_test)
    # print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    gb_model = GradientBoosting()
    gb_model.fit(X_train, y_train)

    y_pred = gb_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

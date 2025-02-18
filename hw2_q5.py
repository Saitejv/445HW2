import numpy as np


def stump_classification_result(feature: int, threshold: float, compare: str, X: np.ndarray):
    """Calculate the classification result of a decision stump.

    Args:
        feature: feature index
        threshold: boundary location
        compare: string in {"<", ">"}
        X: dataset

    Returns:
        predicted labels
    """
    if compare == ">":
        return np.sign(X[:, feature] - threshold)
    else:
        return np.sign(threshold - X[:, feature])


def best_decision_stump(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[int, float, str]:
    """Calculates the optimal decision stump using a brute force method.

    Args:
        w: weights for the training data
        X: training data
        y: training labels

    Returns:
        the optimal (index, threshold, comparison) decision stump
    """
    n, d = X.shape
    best_error = n
    best_feature = None
    best_threshold = None
    best_operator = None
    
    # brute-force search for best decision stump
    for feature in range(d):
        upper_bound = np.max(X[:, feature]) - 0.005  # avoid points on the decision boundary
        lower_bound = np.min(X[:, feature]) + 0.005
        for threshold in np.arange(lower_bound, upper_bound, 0.01):
            for compare in [">", "<"]:
                t = stump_classification_result(feature, threshold, compare, X)
                err = w.T @ np.not_equal(t, y)
                if err < best_error:
                    best_error = err
                    best_feature = feature
                    best_threshold = threshold
                    best_operator = compare
    
    return best_feature, best_threshold, best_operator


def adaboost(X: np.ndarray, labels: np.ndarray, num_iterations: int) -> tuple[list[tuple[int, float, str]], np.ndarray]:
    """Train an AdaBoost ensemble of decision stumps.

    Args:
        X: training data
        labels: training labels
        num_iterations: how many weak learners to train

    Returns:
        a (functions, alpha) tuple where functions is a list of decision stumps and alpha is the weights of
        each stump
    """
    n, d = X.shape
    w = np.ones(n) / n
    alpha = np.zeros(num_iterations)
    functions = []
    for i in range(num_iterations):
        feature, threshold, operator = best_decision_stump(w, X, labels)
        print(f"decision stump {i}: x_{feature} {operator} {threshold}")
        # TODO: implement the rest of the AdaBoost algorithm

    return functions, alpha


def classify(functions: list[tuple[int, float, str]], alpha: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Make predictions from the ensemble of decision stumps.

    Args:
        functions: list of trained decision stumps
        alpha: weights of each decision stump
        X: input data

    Returns:
        predicted {-1, 1} labels
    """
    # TODO: implement the classification algorithm
    pass


if __name__ == '__main__':
    X = np.array([
        [0.30, 0.15],
        [0.10, 0.15],
        [0.12, 0.17],
        [0.30, 0.17],
        [0.20, 0.18],
        [0.22, 0.19],
        [0.45, 0.20],
    ])
    y = np.array([1, 1, 1, -1, 1, -1, -1])
    
    functions, alpha = adaboost(X, y, 3)
    print(functions)
    print(f"alpha: {alpha}")
    
    t = classify(functions, alpha, X)
    print(f"actual labels:    {y}")
    print(f"predicted labels: {t}")
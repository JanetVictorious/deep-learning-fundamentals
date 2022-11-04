import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def cluster_dataset():
    """Generate four clusters for classification."""
    np.random.seed(1)
    m = 800  # Number of examples
    N = int(m / 4)  # Number of points per class
    D = 2  # Number of features
    X = np.zeros((m, D))  # Feature matrix
    y = np.zeros((m, 1), dtype='uint8')  # Label array

    # Generate clusters
    centers = [[-1, -1], [-1, 1], [1, 1], [1, -1]]
    for j in range(4):
        ix = range(N * j, N * (j + 1))
        x, _ = make_blobs(n_samples=N, centers=[centers[j]], cluster_std=0.6)
        X[ix] = x
        y[ix] = j % 2

    X = X.T
    y = y.T

    return X, y


def cluster_dataset_2():
    """Generate two clusters for classification."""
    np.random.seed(1)
    m = 400  # Number of examples
    N = int(m / 2)  # Number of points per class
    D = 2  # Number of features
    X = np.zeros((m, D))  # Feature matrix
    y = np.zeros((m, 1), dtype='uint8')  # Label array

    # Generate clusters
    centers = [[0, 0]]
    for j in range(2):
        ix = range(N * j, N * (j + 1))
        if j == 0:
            x, _ = make_blobs(n_samples=N, centers=centers, cluster_std=0.4)
        else:
            t = np.linspace(0, np.pi , N) + np.random.randn(N) * 0.2  # Theta
            r = 1.1 + np.random.randn(N) * 0.2  # Radius
            x = np.c_[r * np.cos(t), r * np.sin(t)]
        X[ix] = x
        y[ix] = j % 2

    X = X.T
    y = y.T

    return X, y

def cluster_dataset_3():
    """Generate two clusters for classification."""
    np.random.seed(1)
    m = 400  # Number of examples
    N = int(m / 2)  # Number of points per class
    D = 2  # Number of features
    X = np.zeros((m, D))  # Feature matrix
    y = np.zeros((m, 1), dtype='uint8')  # Label array

    # Generate clusters
    centers = [[0, 0]]
    for j in range(2):
        ix = range(N * j, N * (j + 1))
        if j == 0:
            x, _ = make_blobs(n_samples=N, centers=centers, cluster_std=0.4)
        else:
            t = np.linspace(0, 2.0 * np.pi , N) + np.random.randn(N) * 0.2  # Theta
            r = 1.1 + np.random.randn(N) * 0.2  # Radius
            x = np.c_[r * np.cos(t), r * np.sin(t)]
        X[ix] = x
        y[ix] = j % 2

    X = X.T
    y = y.T

    return X, y


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y)


if __name__ == '__main__':
    cluster_dataset()

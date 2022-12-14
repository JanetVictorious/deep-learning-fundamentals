import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import log_loss, f1_score, accuracy_score
import matplotlib.pyplot as plt
import h5py


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
            t = np.linspace(0, np.pi, N) + np.random.randn(N) * 0.2  # Theta
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
            t = np.linspace(0, 2.0 * np.pi, N) + np.random.randn(N) * 0.2  # Theta
            r = 1.1 + np.random.randn(N) * 0.2  # Radius
            x = np.c_[r * np.cos(t), r * np.sin(t)]
        X[ix] = x
        y[ix] = j % 2

    X = X.T
    y = y.T

    return X, y


def cluster_dataset_4():
    """Generate two clusters for classification."""
    np.random.seed(1)
    m = 800  # Number of examples
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
            t = np.linspace(0, 2.0 * np.pi, N) + np.random.randn(N) * 0.5  # Theta
            r = 1.1 + np.random.randn(N) * 0.2  # Radius
            x = np.c_[r * np.cos(t), r * np.sin(t)]
        X[ix] = x
        y[ix] = j % 2

    X = X.T
    y = y.T

    return X, y


def cluster_dataset_5():
    """Generate two clusters for classification."""
    for i in range(2):
        np.random.seed(i + 1)
        if i == 0:
            m = 400  # Number of examples
        else:
            m = 200
        N = int(m / 2)  # Number of points per class
        D = 2  # Number of features
        X = np.zeros((m, D))  # Feature matrix
        y = np.zeros((m, 1), dtype='uint8')  # Label array

        # Generate clusters
        for j in range(2):
            ix = range(N * j, N * (j + 1))
            if j == 0:
                t = np.linspace(5 * np.pi / 4, 9 * np.pi / 4, N) + np.random.randn(N) * 0.2  # Theta
            else:
                t = np.linspace(np.pi / 4, 5 * np.pi / 4, N) + np.random.randn(N) * 0.2  # Theta
            r = 0.55 + np.random.randn(N) * 0.2  # Radius
            x = np.c_[r * np.cos(t), r * np.sin(t)]
            X[ix] = x
            y[ix] = j % 2

        if i == 0:
            X_train = X.T
            y_train = y.T
        else:
            X_test = X.T
            y_test = y.T

    return X_train, y_train, X_test, y_test


def cluster_dataset_6():
    """Generate two clusters for classification."""
    for i in range(2):
        np.random.seed(i + 1)
        if i == 0:
            m = 400  # Number of examples
        else:
            m = 200
        N = int(m / 2)  # Number of points per class
        D = 2  # Number of features
        X = np.zeros((m, D))  # Feature matrix
        y = np.zeros((m, 1), dtype='uint8')  # Label array

        # Generate clusters
        for j in range(2):
            ix = range(N * j, N * (j + 1))
            if j == 0:
                t = np.linspace(-np.pi / 8.0, 9 * np.pi / 8.0, N) + np.random.randn(N) * 0.15  # Theta
            else:
                t = np.linspace(7 * np.pi / 8.0, 17 * np.pi / 8.0, N) + np.random.randn(N) * 0.15  # Theta
            r = 0.8 + np.random.randn(N) * 0.15  # Radius
            if j == 0:
                x = np.c_[r * np.cos(t), r * np.sin(t)]
            elif j == 1:
                x = np.c_[r * np.cos(t) + 0.8, r * np.sin(t)]
            X[ix] = x
            y[ix] = j % 2

        if i == 0:
            X_train = X.T
            y_train = y.T
        else:
            X_test = X.T
            y_test = y.T

    return X_train, y_train, X_test, y_test


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


def load_cat_data():
    train_dataset = h5py.File('data/catvsnocat/train_catvnoncat.h5', 'r')
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])  # Train set features
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])  # Train set labels

    test_dataset = h5py.File('data/catvsnocat/test_catvnoncat.hdf', 'r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])  # Test set features
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])  # Test set labels

    classes = np.array(test_dataset['list_classes'][:])  # List of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def plot_results(model, X_train, y_train, X_test, y_test):
    """Plot results from model."""
    # Plot decision boundary
    plot_decision_boundary(lambda x: model.predict(x.T), X_train, y_train)

    # Performance metrics training set
    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)

    print('Training data preformance:')
    print(f'Accuracy: {accuracy_score(y_train.reshape(-1,), y_pred.reshape(-1,))}')
    print(f'Logloss: {log_loss(y_train.reshape(-1,), y_pred_proba.reshape(-1,))}')
    print(f'F1 score: {f1_score(y_train.reshape(-1,), y_pred.reshape(-1,))}')

    print('Test data preformance:')
    print(f'Accuracy: {accuracy_score(y_test.reshape(-1,), y_pred_test.reshape(-1,))}')
    print(f'Logloss: {log_loss(y_test.reshape(-1,), y_pred_proba_test.reshape(-1,))}')
    print(f'F1 score: {f1_score(y_test.reshape(-1,), y_pred_test.reshape(-1,))}')

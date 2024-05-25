import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    W3 = np.random.rand(10, 10) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    W4 = np.random.rand(10, 10) - 0.5
    b4 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3, W4, b4

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X):
    # Layer 1
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    # Layer 2
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)

    # Layer 3
    Z3 = W3.dot(A2) + b3
    A3 = ReLU(Z3)

    # Layer 4
    Z4 = W4.dot(A3) + b4
    A4 = softmax(Z4)

    return Z1, A1, Z2, A2, Z3, A3, Z4, A4

def one_hot(Y):
    one_hot_Y = np.zeros((Y.max() + 1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, Z3, A3, A4, W2, W3, W4, X, Y):
    one_hot_Y = one_hot(Y)
    dZ4 = A4 - one_hot_Y
    dW4 = (1 / Y.size) * dZ4.dot(A3.T)
    db4 = (1 / Y.size) * np.sum(dZ4, axis=1, keepdims=True)
    dZ3 = W4.T.dot(dZ4) * deriv_ReLU(Z3)
    dW3 = (1 / Y.size) * dZ3.dot(A2.T)
    db3 = (1 / Y.size) * np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = W3.T.dot(dZ3) * deriv_ReLU(Z2)
    dW2 = (1 / Y.size) * dZ2.dot(A1.T)
    db2 = (1 / Y.size) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = (1 / Y.size) * dZ1.dot(X.T)
    db1 = (1 / Y.size) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3, dW4, db4

def update_params(W1, b1, W2, b2, W3, b3, W4, b4, dW1, db1, dW2, db2, dW3, db3, dW4, db4, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W3 -= alpha * dW3
    b3 -= alpha * db3
    W4 -= alpha * dW4
    b4 -= alpha * db4
    return W1, b1, W2, b2, W3, b3, W4, b4

def get_predictions(A4):
    return np.argmax(A4, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2, W3, b3, W4, b4 = init_params()
    accuracy_list = []
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3, Z4, A4 = forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X)
        dW1, db1, dW2, db2, dW3, db3, dW4, db4  = back_prop(Z1, A1, Z2, A2, Z3, A3, A4, W2, W3, W4, X, Y)
        W1, b1, W2, b2, W3, b3, W4, b4 = update_params(W1, b1, W2, b2, W3, b3, W4, b4, dW1, db1, dW2, db2, dW3, db3, dW4, db4, alpha)
        if i % 10 == 0:
            accuracy = get_accuracy(get_predictions(A4), Y)
            accuracy_list.append(accuracy)
            print('Iteration:', i)
            print("Accuracy:", accuracy)
    plt.plot(range(0, iterations, 10), accuracy_list)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Time')
    plt.show()
    return W1, b1, W2, b2, W3, b3, W4, b4

def make_predictions(X, W1, b1, W2, b2, W3, b3, W4, b4):
    _, _, _, _, _, _, _, A4 = forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X)
    predictions = get_predictions(A4)
    return predictions

def test_prediction(index, W1, b1, W2, b2, W3, b3, W4, b4):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3, W4, b4)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    # plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Load data
data = pd.read_csv('./data/train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.

iterations = 10000
alpha = 0.2

W1, b1, W2, b2, W3, b3, W4, b4 = gradient_descent(X_train, Y_train, iterations, alpha)

dev_predictions = make_predictions(X_test, W1, b1, W2, b2, W3, b3, W4, b4)
dev_accuracy = get_accuracy(dev_predictions, Y_test)
print("Development Accuracy: ", dev_accuracy)
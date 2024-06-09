import random
import sqlite3
from io import BytesIO

import numpy as np
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def matrix_multiplication(a: list[list[float]], b: list[list[float]]):
    if len(a[0]) != len(b):
        raise ValueError
    no_rows = len(a)
    no_columns = len(b[0])
    result = [[0 for _ in range(no_columns)] for _ in range(no_rows)]
    for i in range(no_rows):
        for j in range(no_columns):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]

    return result


def matrix_add(a, b):
    no_rows = len(a)
    no_columns = len(a[0])
    result = [[0 for _ in range(no_columns)] for _ in range(no_rows)]
    for i in range(no_rows):
        for j in range(no_columns):
            result[i][j] = a[i][j] + b[i][j]

    return result


def transpose_matrix(a):
    no_rows = len(a)
    no_columns = len(a[0])
    result = [[0 for _ in range(no_rows)] for _ in range(no_columns)]
    for i in range(no_rows):
        for j in range(no_columns):
            result[j][i] = a[i][j]

    return result


class AbstractLayer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


class FullyConnectedLayer(AbstractLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = [[random.random() for _ in range(output_size)] for _ in range(input_size)]
        self.bias = [[random.random()] for _ in range(output_size)]

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = matrix_add(matrix_multiplication(self.input, self.weights), self.bias)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = matrix_multiplication(output_error, transpose_matrix(self.weights))
        weights_error = matrix_multiplication(transpose_matrix(self.input), output_error)

        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                self.weights[i][j] -= learning_rate * weights_error[i][j]

        for i in range(len(self.bias)):
            for j in range(len(self.bias[0])):
                self.bias[i][j] -= learning_rate * output_error[i][j]

        return input_error


class ActivationLayer(AbstractLayer):

    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_arg):
        self.input = input_arg
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return [[float(x) * output_error for x in self.activation_prime(self.input)]]


def relu(x):
    return [max(0, y) for t in x for y in t]


def relu_prime(x):
    return [1 if y > 0 else 0 for t in x for y in t]


def stable_sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (np.exp(x) + 1)


def stable_sigmoid_derivative(x):
    sigmoid_x = stable_sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


def sigmoid(x):
    return [stable_sigmoid(y) for t in x for y in t]


def sigmoid_prime(x):
    return [stable_sigmoid_derivative(y) for t in x for y in t]


def mse(yr, yp):
    s = 0
    for i in range(len(yp)):
        s += (yp[i] - yr[i]) ** 2
    return s / len(yr)


def mse_prime(yr, yp):
    s = 0
    for i in range(len(yp)):
        s += (yp[i] - yr[i])
    return 2 / len(yr) * s


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_arg):
        samples = len(input_arg)
        result = []
        for i in range(samples):
            output = input_arg[i]
            for layer in self.layers:
                layer: AbstractLayer
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    def fit(self, train_input_arg, train_output_arg, epochs=500, learning_rate=0.001):
        samples = len(train_input_arg)
        for i in range(epochs):
            err = 0
            train_input_shuffled = random.sample(train_input_arg, len(train_input_arg))
            for j in range(samples):
                output = train_input_shuffled[j]
                for layer in self.layers:
                    layer: AbstractLayer
                    output = layer.forward_propagation(output)

                if isinstance(train_output_arg[j], float):
                    err += self.loss([train_output_arg[j]], output)
                    error = self.loss_prime([train_output_arg[j]], output)
                else:
                    err += self.loss([train_output_arg[j]], output)
                    error = self.loss_prime([train_output_arg[j]], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err /= samples
            print('epoch %d/%d   error=%f' % (i + 1, epochs, err))


def load_data_from_db(test, target_size=(32, 32)):
    conn = sqlite3.connect('images.db')
    cursor = conn.cursor()
    cursor.execute("SELECT imagine, sepia FROM Imagini WHERE test = ?", (True if test == 1 else False,))
    rows = cursor.fetchall()
    conn.close()

    x = []
    y = []
    for row in rows:
        imagine_blob, sepia = row
        imagine_bytes = BytesIO(imagine_blob)
        imagine = Image.open(imagine_bytes)

        if imagine.mode == "P":
            imagine = imagine.convert("RGBA")
        if imagine.mode != 'RGB':
            imagine = imagine.convert('RGB')

        imagine = imagine.resize(target_size)

        imagine_data = list(imagine.getdata())

        imagine_list = [component / 255.0 for pixel in imagine_data for component in pixel]

        x.append(imagine_list)
        y.append(1 if sepia else 0)

    return x, y


train_input, train_output = load_data_from_db(0)
test_input, test_output = load_data_from_db(1)
train_input = [[x] for x in train_input]
test_input = [[x] for x in test_input]

model = Network()

model.add(FullyConnectedLayer(32 * 32 * 3, 1))
model.add(ActivationLayer(sigmoid, sigmoid_prime))

model.use(mse, mse_prime)

model.fit(train_input, train_output, epochs=10, learning_rate=0.1)

predicted = model.predict(test_input)
print(predicted)
predicted = [1 if x > 0.5 else 0 for p in predicted for x in p]
cm = confusion_matrix(test_output, predicted)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=["normal", "sepia"], yticklabels=["normal", "sepia"])
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

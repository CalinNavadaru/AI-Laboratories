class MyGDRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coefficients = []

    def fit(self, x, y, learning_rate=0.001, no_epochs=1000, batch_size=16):
        self.coefficients = [0.0 for _ in range(len(x) + 1)]
        for epoch in range(no_epochs):
            for i in range(0, len(x), batch_size):
                gradients = self.compute_gradients(x[i:i + batch_size], y[i:i + batch_size])
                for j in range(len(x[0])):
                    self.coefficients[j] -= learning_rate * gradients[j]
                self.coefficients[-1] -= learning_rate * gradients[-1]

        self.intercept_ = self.coefficients[-1]
        self.coefficients = self.coefficients[:-1]

    def eval(self, xi):
        yi = self.coefficients[-1]
        for j in range(len(xi)):
            yi += self.coefficients[j] * xi[j]
        return yi

    def predict(self, x):
        y_computed = [self.eval(xi) for xi in x]
        return y_computed

    def compute_gradients(self, x, y):
        gradients = [0 for _ in range(len(x) + 1)]
        for i in range(len(x)):
            y_computed = self.eval(x[i])
            crt_error = y_computed - y[i]
            for j in range(len(x[0])):
                gradients[j] += crt_error * x[i][j]
            gradients[-1] += crt_error
        return gradients
import os
import csv

from matplotlib import pyplot as plt


def load_data(file_name, input_variable_name, output_variable_name):
    data = []
    data_names = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                data_names = row
            else:
                data.append(row)
            line_count += 1
    selected_variable = data_names.index(input_variable_name)
    input_list = [float(data[i][selected_variable]) for i in range(len(data))]
    selected_output = data_names.index(output_variable_name)
    output_list = [float(data[i][selected_output]) for i in range(len(data))]

    return input_list, output_list


crtDir = os.getcwd()
filePath = os.path.join(crtDir, 'data', 'world-happiness-report-2017.csv')

inputs, outputs = load_data(filePath, 'Economy..GDP.per.Capita.', 'Happiness.Score')
print('in:  ', inputs[:5])
print('out: ', outputs[:5])
#%%
def plot_histogram(data, variable_name):
    _ = plt.hist(data, 10)
    plt.title("Histogram of " + variable_name)
    plt.show()


plot_histogram(inputs, 'Family')
plot_histogram(outputs, 'Happiness Score')
#%%
plt.plot(inputs, outputs, 'ro')
plt.xlabel('Family')
plt.ylabel('happiness')
plt.title('Family vs. happiness')
plt.show()
#%%
import random

indexes = [i for i in range(len(inputs))]
train_sample = random.sample(indexes, k=int(0.8 * len(indexes)))
validation_sample = [i for i in indexes if not i in train_sample]

train_inputs = [inputs[i] for i in train_sample]
train_outputs = [outputs[i] for i in train_sample]

validation_inputs = [inputs[i] for i in validation_sample]
validation_outputs = [outputs[i] for i in validation_sample]

plt.plot(train_inputs, train_outputs, 'ro', label='training data')  #train data are plotted by red and circle sign
plt.plot(validation_inputs, validation_outputs, 'g^',
         label='validation data')  #test data are plotted by green and a triangle sign
plt.title('train and validation data')
plt.xlabel('GDP capita')
plt.ylabel('happiness')
plt.legend()
plt.show()
regressor = MyGDRegression()

train_inputs = [[d] for d in train_inputs]
regressor.fit(train_inputs, train_outputs)

validation_inputs = [[d] for d in validation_inputs]
predicted = regressor.predict(validation_inputs)
print(predicted)
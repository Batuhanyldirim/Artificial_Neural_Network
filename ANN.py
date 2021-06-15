"""
This code gets %89 accuracy on MNIST dataset to label hadwritten digits which coded without using any libraries except pandas or numpy
"""

import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


from google.colab import drive
drive.mount('/content/gdrive')

data = pd.read_csv("/content/gdrive/MyDrive/My Desktop/Erasmus/AI/assignment5.csv")

data.head()

data = data.to_numpy()

#Preprocessing

x = []
y = []
for i in data:
  x.append(np.divide(i[1:], 255))
  y.append(i[0])

train_size = int(len(x)*0.7)
validation_size = int(len(x)*0.1)
test_size = len(x)-1

x_train = x[:train_size]
y_train = y[:train_size]

x_validation = x[train_size:train_size+validation_size]
y_validation = y[train_size:train_size+validation_size]

x_test = x[train_size+validation_size:train_size+validation_size+test_size]
y_test = y[train_size+validation_size:train_size+validation_size+test_size]

neuron_num = [784,256,10]
np.random.seed(1)
layers = []
biases = []

l_rate = 0.01
correct_training = 0
correct_validate = 0
correct_test = 0



layers = []
biases = []

batch_size = 10
deltas = np.copy(layers) * 0
biases_change = np.copy(biases) * 0
accuracy = 0


global max_prob
max_prob=0


#Functions

def sigmoid_func(activation):
  return 1.0 / (1.0 + np.exp(-activation))

def sigmoid_derivative(output):
  return output*(1.0-output)


def softmax_function(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum(axis=0)

def softmax_derivative(x):
    return np.ones(len(x))


def encode(output):
    global max_prob
    max_prob = 0
    prediction = 0
    # Select the number with highest probability
    for j in range(len(output)):
        if output[j] > max_prob:
            max_prob = output[j]
            prediction = j

    return prediction

#Train
for i in range(2):
    layers.append(2*np.random.rand(neuron_num[i], neuron_num[i+1]) - 1)
    biases.append(2*np.random.random(neuron_num[i+1])-1)

batch_size = 10
deltas = np.copy(layers) * 0
biases_change = np.copy(biases) * 0
accuracy = 0

for i in range(len(x_train)):
    
    instance = x_train[i]
    inp = [instance]

    
    inp.append(sigmoid_func(np.add(np.dot(inp[-1], layers[0]), biases[0])))
    inp.append(softmax_function(np.add(np.dot(inp[-1], layers[-1]), biases[-1])))

    if encode(inp[-1]) == y_train[i]:
        correct_training += 1

    output = np.zeros(10)
    output[y_train[i]] = 1
    #errors
    errors = [(output - inp[-1]) * softmax_derivative(inp[-1])]
    errors.insert(0,  np.dot(errors[0], layers[1].T) * sigmoid_derivative(inp[1]))


    for k in range(2):
        deltas[k] += (np.dot(inp[k][:,None], errors[k][None,:]) * l_rate)
        biases_change[k] += errors[k] * l_rate

    if (i % batch_size == 0 and i != 0) or i == train_size-1:
        layers += deltas
        biases += biases_change
        deltas = deltas * 0
        biases_change = biases_change * 0

    if i % 1000 == 0 and i != 0:
        print("Accuracy:",correct_training/i)

#Validation
for i in range(len(x_validation)):
    instance = x_validation[i]
    inp = [instance]
    
    inp.append(sigmoid_func(np.dot(inp[-1], layers[0]) + biases[0]))
    inp.append(softmax_function(np.dot(inp[-1], layers[-1]) + biases[-1]))

    max_prob = 0
    if y_validation[i] == encode(inp[-1]):
        correct_validate += 1


accuracy = correct_validate / validation_size
print("Validation accuracy:",accuracy)

#Test

for i in range(int(len(x)*0.8), len(x)-1):
    
    instance = x[i]
    
    inp = [instance]
    
    inp.append(sigmoid_func(np.dot(inp[-1], layers[0]) + biases[0]))
    inp.append(softmax_function(np.dot(inp[-1], layers[-1]) + biases[-1]))

    max_prob = 0

    """print("label:",labels[i])
    print("Prediction:", encode(inp[-1]))
    img = x.reshape(28,28)
    plt.imshow(img, cmap='gray')
    plt.show()
    input()"""


    if y[i] == encode(inp[-1]):
        correct_test += 1


accuracy = correct_test / ( len(x)-1 - int(len(x)*0.8))
print("Total accuracy is:",accuracy)

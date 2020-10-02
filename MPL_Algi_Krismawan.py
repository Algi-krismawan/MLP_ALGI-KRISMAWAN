import math
import numpy as np


# activation function
def sig(x):
    y = 1 / (1 + math.e ** (-x))
    return y


# training data
x = np.array([[1, 0, 1],
              [1, 1, 1],
              [0, 1, 1],
              [0, 0, 1]])
# target value
y = np.array([[1],
              [1],
              [0],
              [0]])

# weight initialization
np.random.seed(1)
w1 = 2 * np.random.random((3, 4)) - 1
w2 = 2 * np.random.random((4, 1)) - 1

# training
for j in range(1000):
    # synapses
    syn1 = sig(np.dot(x, w1))
    syn2 = sig(np.dot(syn1, w2))

    # backpropagation
    syn2_error = y - syn2
    syn2_delta = syn2_error * sig(syn2)
    w2 += syn1.T.dot(syn2_delta)

    syn1_error = syn2_delta.dot(w2.T)
    syn1_delta = syn1_error * sig(syn1)
    w1 += x.T.dot(syn1_delta)

# output
print(syn2)
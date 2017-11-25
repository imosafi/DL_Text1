import xor_data
import train_mlp1
import mlp1
import numpy as np
import random

dataList = []
for label, data in xor_data.data:
    dataList.append([label, np.array(data)])

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.
    """
    # give different values than 0
    W = np.random.uniform(-1, 1, (in_dim, hid_dim))
    U = np.random.uniform(-1, 1, (hid_dim, out_dim))
    bw = np.random.uniform(0.1, 0.1, (hid_dim))
    bu = np.random.uniform(0.1, 0.1, (out_dim))

    return [U, W, bu, bw]

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        if label == mlp1.predict(features, params):
            good = good + 1
        else:
            bad = bad + 1
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    U, W, bu, bw = params
    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = features
            y = label
            # grads is [gW, gbw, gU, gbu]
            loss, grads = mlp1.loss_and_gradients(x, y, [U, W, bu, bw])
            cum_loss += loss
            W = W - learning_rate * grads[0]
            bw = bw - learning_rate * grads[1]
            U = U - learning_rate * grads[2]
            bu = bu - learning_rate * grads[3]
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, [U, W, bu, bw])
        dev_accuracy = accuracy_on_dataset(dev_data, [U, W, bu, bw])
        print I+1, train_loss, train_accuracy, dev_accuracy
    return params

params = create_classifier(2, 2, 2)
trainedParams = train_classifier(dataList, dataList, 200, 0.1, params)


import loglinear as ll
import random
import utils
import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def feats_to_vec(features):
    x = np.zeros(600)
    num_of_features = len(features)
    for item in features:
        if item in utils.F2I:
            x[utils.F2I[item]] = x[utils.F2I[item]] + 1.0 / num_of_features
    return x

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        if utils.L2I[label] == ll.predict(feats_to_vec(features), params):
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
    W, b = params
    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = utils.L2I[label]
            loss, grads = ll.loss_and_gradients(x, y, [W, b])
            cum_loss += loss
            W = W - learning_rate * grads[0]
            b = b - learning_rate * grads[1]
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, [W, b])
        dev_accuracy = accuracy_on_dataset(dev_data, [W, b])
        print I+1, train_loss, train_accuracy, dev_accuracy
    return params

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    train_data = utils.TRAIN
    dev_data = utils.DEV

    learning_rate = 0.2
    num_iterations = 20
    out_dim = len(utils.L2I)
    in_dim = len(utils.F2I)

    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)


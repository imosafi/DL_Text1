import numpy as np
from math import log,sqrt

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def classifier_output(x, params):
    # YOUR CODE HERE
    i = 0
    value = x
    while i < len(params) - 1:
        value = np.tanh(np.dot(value, params[i][0]) + params[i][1])
        i += 1
    return softmax(np.dot(value, params[-1][0]) + params[-1][1])


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    gradients_values = []
    y_label = np.zeros(params[-1][1].shape)
    y_label[y] = 1

    y_pred = classifier_output(x, params)
    loss = -log(y_pred[y])

    z_l, a_l = calc_params_for_gradients(x, params)
    g_last_b = y_pred - y_label

    gradients_values.insert(0, (calc_weight_matrix_grad(a_l[-2], g_last_b), g_last_b))


    i = len(params) - 2
    while i > -1:
        gb = calc_weight_vector_grad(gradients_values[0][1], params[i+1][0], a_l[i+1])
        gw = calc_weight_matrix_grad(z_l[i], gb)
        gradients_values.insert(0, [gw, gb])
        i -= 1
    return loss, gradients_values

def calc_weight_matrix_grad(x, gb):
    return np.array([x]).transpose().dot([gb])


def calc_weight_vector_grad(grad, mat, a):
    return grad.dot(mat.transpose()) * (1 - np.power(a, 2))


def calc_params_for_gradients(x, params):
    z_l = [x]
    a_l = [x]

    i = 0
    while i < len(params):
        z_l.append(np.array(z_l[-1]).dot(params[i][0]) + params[i][1])
        a_l.append(np.tanh(z_l[-1]))
        i += 1
    return z_l, a_l


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.

    Assume a tanh activation function between all the layers.

    return:
    a list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    init_value = calc_init_values(dims[0], dims[-1])
    for index, val in enumerate(dims):
        if (index != len(dims) - 1):
            params.append([np.random.uniform(-init_value, init_value, (dims[index], dims[index + 1])),
                           np.zeros(dims[index + 1])])
    return params

def calc_init_values(dim_im, dim_out):
    return float(sqrt(6)) / sqrt(dim_im + dim_out)
import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def softmax(x):
    e_x = np.exp(x - np.max(x))
    bla = e_x / e_x.sum()
    return bla

def classifier_output(x, params):
    # YOUR CODE HERE.

    # W,b = params
    # probs = softmax(np.dot(x, W) + b)

    U, W, bu, bw = params

    l1_output = np.tanh(np.dot(x, W) + bw)
    l2_output = np.dot(l1_output, U) + bu
    probs = softmax(l2_output)
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def calc_weight_matrix_grad(x, gb):
    return np.array([x]).transpose().dot([gb])

def loss_and_gradients(x, y, params):
    # YOU CODE HERE
    U, W, bu, bw = params
    y_label = np.zeros(bu.shape)
    y_label[y] = 1

    y_pred = classifier_output(x, params)
    loss = -np.log(y_pred[y])

    z = np.array(x).dot(W) + bw
    a = np.tanh(z)
    gbu = y_pred - y_label
    gU = calc_weight_matrix_grad(a, gbu)
    # gU = np.array([a1]).transpose().dot([gbu])
    gbw = gbu.dot(U.transpose()) * (1 - np.power(a, 2))
    gW = calc_weight_matrix_grad(x, gbw)
    # gW = np.array([x]).transpose().dot([gbw])
    return loss, [gW, gbw, gU, gbu]

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

if __name__ == '__main__':
    from grad_check import gradient_check

    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    U, W, bu, bw = create_classifier(10, 5, 2)
    output = classifier_output(x, [U, W, bu, bw])

    t = 8
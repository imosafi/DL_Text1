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

    W1, W2, b1, b2 = params

    l1_output = np.tanh(np.dot(x, W1) + b1)
    l2_output = np.dot(l1_output, W2) + b2
    probs = softmax(l2_output)

    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    # YOU CODE HERE
    return None

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.
    """
    # give different values than 0
    W1 = np.zeros((in_dim, hid_dim))
    W2 = np.zeros((hid_dim, out_dim))
    b1 = np.zeros(hid_dim)
    b2 = np.zeros(out_dim)

    return [W1, W2, b1, b2]

if __name__ == '__main__':
    from grad_check import gradient_check

    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    W1, W2, b1, b2 = create_classifier(10, 5, 2)
    output = classifier_output(x, [W1, W2, b1, b2])

    t = 8
import numpy as np
import matplotlib.pyplot as plt

#NOT USED
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return(s)



def initialize_parameters(layer_dims):
    parameters={}
    L = len(layer_dims)

    for l in range(1, L):
        #Sets inital parameters with He initialization *np.sqrt(2/previous layer dimension)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2/layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return(parameters)


def linear_forward(A,W,b):
    # A (dim_prev, m)
    # W (dim, dim_prev)
    Z =(np.dot(W,A)+b)
    cache = (A,W,b)
    return(Z,cache)

def leaky_relu(Z):
    a = np.maximum(Z,0.00001*Z)
    return(a)

def linear_activation_forward(A_prev,W,b):
    Z, linear_cache = linear_forward(A_prev,W,b)
    A = leaky_relu(Z)
    cache = (linear_cache,Z)
    return(A,cache)

def drop_neurons(A,keep_prob):
    D = np.random.rand(A.shape[0],A.shape[1])
    D = (D < keep_prob)
    A = A*D/keep_prob
    return(A,D)

def L_model_forward(X, parameters,keep_prob):
    caches = []
    A = X
    dropout_patterns = []
    L = len(parameters) // 2

    # Propogates forward using only relu activation functions
    for l in range(1, L+1):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)])

        if l < L:
            A,D = drop_neurons(A,keep_prob)
            dropout_patterns.append(D)

        caches.append(cache)
    # AL is the output of the final layer
    AL = A
    return AL, caches ,dropout_patterns

def compute_cost(AL,Y):
    # Computes cost as sum of squared differences / 2*m
    m = Y.shape[1]
    cost = np.sum(np.square(AL - Y)) / (2 * m)
    cost = np.squeeze(cost)
    return(cost)


def linear_backward(dZ,cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m

    dA_prev = np.dot(W.T, dZ)
    return (dA_prev, dW, db)

def leaky_relu_backward(dA,Z):
    Z[Z<0]=0.00001
    Z[Z>0]=1
    return(dA*Z)


def linear_activation_backward(dA,cache,D=1,keep_prob=1):
    linear_cache, Z = cache
    dA = dropout_backwards(dA,D,keep_prob)
    dZ = leaky_relu_backward(dA, Z)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def dropout_backwards(dA, D, keep_prob):
    dA = dA*D/keep_prob
    return(dA)

def L_model_backward(AL, Y, caches, dropout_patterns,keep_prob):
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]

    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Change in cost with respect to output of final layer (for squared difference cost function)
    dAL = (AL-Y)

    current_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward(dAL, current_cache)

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = \
            linear_activation_backward(grads["dA" + str(l + 1)], current_cache,dropout_patterns[l], keep_prob)

        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def L_layer_model(X, Y, layers_dims, num_iterations, learning_rate=0.001, keep_prob=0.87, print_cost=False):
    # MAIN NEURAL NETWORK FUNCTION
    costs = []  # keep track of cost
    parameters = initialize_parameters(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        AL, caches, dropout_patterns = L_model_forward(X, parameters,keep_prob)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches, dropout_patterns, keep_prob)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 200 == 1:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 5 == 1:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return(parameters)

def predict(parameters,X):

    AL, caches,ignore = L_model_forward(X,parameters,keep_prob=1)
    return(AL)


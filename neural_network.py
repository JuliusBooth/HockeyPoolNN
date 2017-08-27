from Organize_Stats import get_data
import numpy as np
import matplotlib.pyplot as plt


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y,names = get_data()
# x.shape = (dim,m)     y.shape = (1,m)



def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return(s)



def initialize_parameters(layer_dims):
    parameters={}
    L = len(layer_dims)

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return(parameters)


def linear_forward(A,W,b):
    # A (dim_prev, m)
    # W (dim, dim_prev)
    Z =(np.dot(W,A)+b)
    cache = (A,W,b)
    return(Z,cache)

def relu(Z):
    a = np.maximum(Z,0.00001*Z)
    return(a,[Z])

def linear_activation_forward(A_prev,W,b):
    Z, linear_cache = linear_forward(A_prev,W,b)
    A, activation_cache = relu(Z)
    cache = (linear_cache,activation_cache)
    return(A,cache)


def L_model_forward(X, parameters):
    caches = []
    A = X

    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)])
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)])
    caches.append(cache)

    return AL, caches

def compute_cost(AL,Y):
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

def relu_backward(dA,activation_cache):
    Z = activation_cache[0]
    Z[Z<0]=0.00001
    Z[Z>0]=1
    return(dA*Z)


def linear_activation_backward(dA,cache):
    linear_cache, activation_cache = cache

    dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL


    dAL = (AL-Y)

    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache)


    for l in reversed(range(L - 1)):

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache)

        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]


    return parameters


def L_layer_model(X, Y, layers_dims, num_iterations, learning_rate=0.0003, print_cost=False):

    costs = []  # keep track of cost
    parameters = initialize_parameters(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 1:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 5 == 1:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

def predict(parameters,X):
    AL, caches = L_model_forward(X,parameters)
    return(AL)

layer1_dims = (train_set_x_orig.shape[0])
layers_dims = [layer1_dims, 30,20,10,5, 1]
parameters = L_layer_model(train_set_x_orig, train_set_y, layers_dims, num_iterations=3301, print_cost=True)

Y_prediction_test = predict(parameters, test_set_x_orig)
Y_prediction_train = predict(parameters, train_set_x_orig)

# Print train/test Errors
print("train accuracy: {}".format(np.mean(np.abs(Y_prediction_train - train_set_y))))
print("test accuracy: {}".format(np.mean(np.abs(Y_prediction_test - test_set_y))))




predicted_points = np.concatenate((names.T,Y_prediction_test.T,test_set_y.T),axis=1)
print(predicted_points)

pny = predicted_points
next_year = pny[pny[:,2].argsort(axis=0)[::-1]]
with open("predictions2.txt","w") as f:
    for row in next_year:
        for x in row:

            f.write(str(x))
            f.write("\t")
        f.write("\n")



#######
def propagate(w, b, X, y):
    m = X.shape[1]

    A = (np.dot(w.T,X)+b)

    cost = np.sum(np.square(A - y))/(2*m)
    cost = np.squeeze(cost)

    dw = (1 / m) * np.dot(X, (A - y).T)
    db = (1 / m) * np.sum(A - y)

    grads = {"dw": dw,
             "db": db}

    return(grads, cost)



#########
def optimize(w, b, X, y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            print(w,b)

    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}

    return(params, grads, costs)



#########

def predict(w,b,X):
    m = X.shape[1]

    #names = X[:,:2]
    #X = X[:,2:]


    w = w.reshape(X.shape[0], 1)


    A = (np.dot(w.T, X) + b)
    return(A)



#########
def model(X_train, Y_train, X_test, Y_test, num_iterations = 1000, learning_rate = 0.00005, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)


    # Print train/test Errors
    print("train accuracy: {}".format(np.mean(np.abs(Y_prediction_train - Y_train))))
    print("test accuracy: {}".format(np.mean(np.abs(Y_prediction_test - Y_test))))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return (d)



d = model(train_set_x_orig,train_set_y,test_set_x_orig,test_set_y)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

predicted_points = np.concatenate((names.T,d["Y_prediction_test"].T,test_set_y.T),axis=1)
print(predicted_points)

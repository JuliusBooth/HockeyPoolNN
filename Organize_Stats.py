import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from simplify_data import simplify_years


def get_data(predict_ny):
    # Organizes data into train and test sets
    if predict_ny:
        data,test = simplify_years(True)
    else: data = simplify_years()

    # Filters out players who suck because stars probably don't develop like plugs
    data = data.loc[(data["P"] > 30) | (data["P_prev"] > 30) | (data["P_2prev"] > 30)]

    data["P/82_next"] = 82 * data["P_next"] / data["GP_next"]

    if predict_ny:
        test = test.loc[(test["P"] > 30) | (test["P_prev"] > 30) | (test["P_2prev"] > 30)]
        train = data
        pred_val = "P"
    else:
        # Change random state to get new test data
        train,test = train_test_split(data,test_size=0.2,random_state=2)
        pred_val = "P/82_next"
    return(train,test,pred_val)


def get_inputs(cols,predict_ny=False):
    # Normalizes train and test data

    train, test, pred_val = get_data(predict_ny)
    train_X = train.as_matrix(columns = cols)
    train_y = train.as_matrix(columns=["P/82_next"])
    names = test.as_matrix(columns=["Player","Season"])

    test_X = test.as_matrix(columns=cols)
    test_y = test.as_matrix(columns=[pred_val])

    norm_train_x,norm_test_x = normalized(train_X.T,test_X.T)

    return(norm_train_x, train_y.T, norm_test_x, test_y.T, names.T)


def normalized(X_train,X_test):
    m=np.mean(X_train, axis=1,keepdims=True)
    std = np.std(X_train, axis=1,keepdims=True)
    norm_train_x = (X_train-m)/std
    norm_test_x = (X_test-m)/std
    return(norm_train_x,norm_test_x)






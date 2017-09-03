from Organize_Stats import get_inputs
import numpy as np
from neural_network import L_layer_model, predict

def save_predictions(Y_test_actual,Y_prediction_test,names):
    # Writes predictions to .txt file in order of predicted points high to low
    # Second number is either actual points/82 player had
    # or total points from 2016-17 if predicting 2017-18
    predicted_points = np.concatenate((names.T,Y_prediction_test.T,Y_test_actual.T),axis=1)
    print(predicted_points)

    next_year = predicted_points[predicted_points[:,(names.shape[0])].argsort(axis=0)[::-1]]
    with open("predictions2.txt","w") as f:
        f.write("NAME OF PLAYER\tAGE\tSEASON\tPREDICTED POINTS\tACTUAL POINTS")
        f.write("\n")
        for row in next_year:
            for x in row:
                f.write(str(x))
                f.write("\t")
            f.write("\n")


def run_network(save=True):
    # THE MAIN FUNCTION. SET ALL HYPER-PARAMETERS HERE.

    # Set predict_ny to True if you want to produce predictions for one year.
    predict_ny = False
    # years_ago is the number of seasons ago you want to use to make future predictons for. years_ago = 1 = 2016/17
    years_ago = 2

    # These are the input categories. It's important to list actual categories or NaNs are added to data and network breaks.
    input_categories = ["GP","TOI","G","A1","A","P","iCF","iFF","iSF","iSCF","ixG","iFOW",
        "Avg.DIST","D?","Age","GP_prev","G_prev","P_prev","iCF_prev","ixG_prev",
                        "GP_2prev","G_2prev","P_2prev", "iCF_2prev","ixG_2prev",
                        "GP_3prev","G_3prev","P_3prev","iCF_3prev","ixG_3prev"]

    # Gets all the data.
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y,names = get_inputs(input_categories,years_ago,predict_ny )
    # x.shape = (dim,m)     y.shape = (1,m)

    layer1_dims = (train_set_x_orig.shape[0])

    # Set the dimensions of network here

    layers_dims = [layer1_dims,30,30, 1]


    # Set the learning rate and # of iterations here.
    # Learning rate seems to need to go down the more layers there are.
    # beta1 controls the how much momentum the gradient has. Closer to 1 =  more momentum. Set to 0 to turn off.
    # beta2 slightly more complicated but it leads to gradient not getting out of control. Leave as is. Set to 0 to turn off.
    # keep_prob controls dropout rate. Lower values lead to less overfitting. Set to 1 to turn off.
    # However if there are layers with few neurons do not allow keep_prob to be too low.
    # Don't want to lose all neurons in a layer
    parameters = L_layer_model(train_set_x_orig, train_set_y, layers_dims,
                               learning_rate=0.01, beta1=0.9, beta2=0.999, num_iterations=4002, keep_prob=0.94,
                               print_cost=True)


    Y_prediction_test = predict(parameters, test_set_x_orig)
    Y_prediction_train = predict(parameters, train_set_x_orig)

    # Print train/test Errors
    # The average absolute difference between predicted points and actual points
    # GOAL RIGHT NOW to get UNDER 9 for test set
    print("train accuracy: {}".format(np.mean(np.abs(Y_prediction_train - train_set_y))))
    print("test accuracy: {}".format(np.mean(np.abs(Y_prediction_test - test_set_y))))

    if save:
        save_predictions(test_set_y,Y_prediction_test,names)



if __name__== "__main__":
    run_network(True)
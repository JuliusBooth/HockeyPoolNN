import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import numpy as np
from simplify_data import simplify_years

'''
def get_year_pairs():
    _2015 = pd.read_excel("/Users/juliusbooth/PycharmProjects/HockeyPool2017/NHL2014-15.xls")
    _2016 = pd.read_excel("/Users/juliusbooth/PycharmProjects/HockeyPool2017/NHL2015-16.xls")
    _2017 = pd.read_excel("/Users/juliusbooth/PycharmProjects/HockeyPool2017/NHL2016-17.xls", skiprows=2)

    _15_16 = pd.merge(_2015, _2016, on=["Last Name", "First Name"])

    _16_17 = pd.merge(_2016, _2017, on=["Last Name", "First Name"])
    _16_17 = _16_17.fillna(0)
    _16_17["D?"] = np.where(_16_17["Pos"] == "D", 1, 0)
    _2017["Age"] = 2017-(_2017["Born"]).str.slice(0,4).astype(int)
    _2017 = _2017.fillna(0)
    _2017["D?"] = np.where(_2017["Position"] == "D", 1, 0)
    return(_16_17,_2017)
'''

def get_data():
    #Returns train_x, train_y, test_x, test_y
    #data,this_year = get_year_pairs()
    '''
        train_X = data.as_matrix(columns=["GP_x", "G_x", "A_x", "Sh", "Age", "Misses", "Shifts_x", "Passes", "ShotDist", "D?"])
        train_y = data.as_matrix(columns=["PTS_y"])
        names = this_year.as_matrix(columns=["First Name", "Last Name"])
        test_X = this_year.as_matrix(columns=["GP","G","A","iSF","Age","iMiss","Shifts","Pass","sDist","D?"])
        test_y = this_year.as_matrix(columns=["PTS"])
    '''


    years = simplify_years()
    data=pd.concat(years)
    data=data.fillna(0)
    train,test = train_test_split(data,test_size=0.2,random_state=1)

    train_X = train.as_matrix(columns = ["GP","TOI","G","A1","A","P","iCF","iFF","iSF","ixG","iFOW","Avg.DIST","D?"])
    train_y = train.as_matrix(columns=["P_y"])
    names = test.as_matrix(columns=["Player","Season"])
    test_X = test.as_matrix(columns=["GP","TOI","G","A1","A","P","iCF","iFF","iSF","ixG","iFOW","Avg.DIST","D?"])
    test_y = test.as_matrix(columns=["P_y"])


    norm_train_x,norm_test_x = normalized(train_X.T,test_X.T)

    return(norm_train_x, train_y.T, norm_test_x, test_y.T, names.T)


def normalized(X_train,X_test):
    m=np.mean(X_train, axis=1,keepdims=True)
    std = np.std(X_train, axis=1,keepdims=True)
    norm_train_x = (X_train-m)/std
    norm_test_x = (X_test-m)/std
    return(norm_train_x,norm_test_x)


get_data()



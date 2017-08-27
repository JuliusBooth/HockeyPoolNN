import pandas as pd
import numpy as np



def get_years():
    years = []
    for i in range(20072008,20162017,10001):
        years.append(pd.read_csv( str(i)  + "NHLCounts.csv"))
    #years.append(pd.read_excel("/Users/juliusbooth/PycharmProjects/HockeyPool2017/NHL2016-17.xls", skiprows=2))
    return (years)

def simplify(year):

    year["D?"] = np.where(year["Position"] == "D", 1, 0)
    return(year)
    #year = year[["First Name","Last Name","GP","G","A","A1","Pts","Sh","Age","Misses","Shifts","Passes","ShotDist","D?"]]

def add_next_year_points(year,nextyear):

    nextyear_points = nextyear[["Player","P"]]
    combined = pd.merge(year, nextyear_points, on=["Player"])
    return(combined)

def simplify_years():
    years = get_years()
    data=[]
    for i in range(len(years)-1):
        years[i] = simplify(years[i])
        years[i] = add_next_year_points(years[i],years[i+1])
        data.append(years[i])
    return(years)

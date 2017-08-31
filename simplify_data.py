import pandas as pd
import numpy as np
import json


def get_years():
    years = []
    for i in range(20072008,20162017,10001):
        years.append(pd.read_csv("count_data_files/" + str(i)  + "NHLCounts.csv"))

    years.append(pd.read_csv("count_data_files/NHL2016-17.csv"))
    return (years)

def simplify(year):
    #adds a binary column indicating whether player is a defenseman or not
    year["D?"] = np.where(year["Position"] == "D", 1, 0)
    return(year)

def add_next_year_info(year,nextyear):
    #adds the points and games played the player had next season
    nextyear_info = nextyear[["Player","P","GP"]]
    combined = pd.merge(year, nextyear_info, on=["Player"],suffixes=["","_next"])
    combined["P/82_next"] = 82 * combined["P_next"] / combined["GP_next"]
    return(combined)

def add_previous_season_info(year,prevyear):
    #adds stats from the previous season
    prevyear_points = prevyear[["Player","GP","G","P","iCF","ixG"]]
    combined = pd.merge(year, prevyear_points, on=["Player"],suffixes=["","_prev"],how="left")
    return (combined)

def add_2prev_season_info(year,ppyear):
    # adds stats from two seasons ago
    ppyear_points = ppyear[["Player","GP","G","P","iCF","ixG"]]
    combined = pd.merge(year, ppyear_points, on=["Player"],suffixes=["","_2prev"],how="left")
    return (combined)
def add_3prev_season_info(year,p3year):
    # adds stats from two seasons ago
    p3year_points = p3year[["Player","GP","G","P","iCF","ixG"]]
    combined = pd.merge(year, p3year_points, on=["Player"],suffixes=["","_3prev"],how="left")
    return (combined)

def add_ages(year,date):
    # adds player ages in years
    age=[]
    birthdates = get_birthdates()
    for row in year["Player"]:
        if row in birthdates:
            age.append(date-int(birthdates[row]))
        else:
            age.append(0)
    year["Age"] = age
    return(year)

def get_birthdates():
    with open("player_ages.json") as f:
        data = json.load(f)
    return(data)

def simplify_years(years_ago,predict_next_year=False):
    #gets the data
    years = get_years()

    #adds the necessary modifications
    years = [simplify(years[i]) for i in range(len(years))]
    years = [add_ages(years[i],2007+i) for i in range(len(years))]
    for i in range(1,len(years)):
        years[i] = add_previous_season_info(years[i],years[i-1])
    for i in range(2, len(years)):
        years[i] = add_2prev_season_info(years[i], years[i - 2])
    for i in range(3, len(years)):
        years[i] = add_3prev_season_info(years[i], years[i - 3])
    for i in range(len(years)-1):
        years[i] = add_next_year_info(years[i],years[i+1])

    # stacks all dataframes except last years into one dataframe and get replace NaNs with 0s
    data = pd.concat(years[:-years_ago])

    data = data.fillna(0)

    # return data (+ last years stats if predict_next_year is True)
    if predict_next_year:
        prediction = years[-years_ago].fillna(0)
        return (data,prediction)
    else: return(data)




# Supervised learning models to predicting football matches outcomes

> **Project developed by:**\
> ([Martim Silva](https://github.com/motapinto))\
> ([Luís Ramos](https://github.com/luispramos))\
> ([Francisco Gonçalves ](https://github.com/kiko-g))
>
> **Any problems?**\
> Start an Issue please.

**Disclaimer** - This repository was created for educational purposes and we do not take any responsibility for anything related to its content. You are free to use any code or algorithm you find, but do so at your own risk.


### Notebook by [Martim Pinto da Silva](https://github.com/motapinto), [Luis
Ramos](https://github.com/luispramos), [Francisco
Gon├ºalves](https://github.com/kiko-g)
#### Supported by [Luis Paulo Reis](https://web.fe.up.pt/~lpreis/)
#### [Faculdade de Engenharia da Universidade do
Porto](https://sigarra.up.pt/feup/en/web_page.inicial)

#### It is recommended to [view this notebook in
nbviewer](https://nbviewer.ipython.org/github.com/motapinto/football-
classification-
predications/blob/master/src/Supervised%20Learning%20Models.ipynb) for the best
overall experience
#### You can also execute the code on this notebook using [Jupyter
Notebook](https://jupyter.org/) or [Binder](https://mybinder.org/)(no local
installation required)

## Table of contents
1. * [Introduction](#Introduction)
2. * [Required libraries and models](#Required-libraries-and-models)
    - [Libraries](#Libraries)
    - [Models](#Models)
3. * [The problem domain](#The-problem-domain)
4. * [Step 1: Data analysis](#Step-1:-Data-analysis)
    - [Extracting data from the database](#Extracting-data-from-the-database)
    - [Matches](#Matches)
    - [Team Stats - Team Attributes](#Team-Stats---Team-Attributes)
    - [Team Stats - Shots](#Team-Stats---Shots)
    - [Team Stats - Possession](#Team-Stats---Possession)
    - [Team Stats - Crosses](#Team-Stats---Crosses)
    - [FIFA data](#FIFA-data)
    - [Joining all features](#Joining-all-features)
5. * [Step 2: Classification & Results
Interpretation](#Step-2:-Classification-&-Results-Interpretation)
    - [Training and Evaluating Models](#Training-and-Evaluating-Models)
    - [The basis](#The-basis)
    - [KNN](#KNN)
    - [Decision Tree](#Decision-Tree)
    - [SVC](#SVC)
    - [Naive Bayes](#Naive-Bayes)
    - [Gradient Boosting](#Gradient-Boosting)
    - [Neural Network](#Neural-Network)
    - [Deep Neural Network](#Deep-Neural-Network)
6. * [Conclusion](#Conclusion)
    - [What did we learn](#What-did-we-learn)
    - [Choosing best model](#Choosing-best-model)
7. * [Resources](#Resources)

## Introduction
[go back to the top](#Table-of-contents)

In the most recent years there's been a major influx of data. In response to
this situation, Machine Learning alongside the field of Data Science have come
to the forefront, representing the desire of humans to better understand and
make sense of the current abundance of data in the world we live in.

In this notebook, we look forward to use Supervised Learning models to harness a
dataset of around 25k football matches in order to be able to predict the
outcome of other matchups according to a set of classes (win, draw, loss, etc.)

## Required libraries and models
[go back to the top](#Table-of-contents)

### Libraries
If you don't have Python on your computer, you can use the [Anaconda Python
distribution](http://continuum.io/downloads) to install most of the Python
packages you need. Anaconda provides a simple double-click installer for your
convenience.

This notebook uses several Python packages that come standard with the Anaconda
Python distribution. The primary libraries that we'll be using are:

**NumPy**: Provides a fast numerical array structure and helper functions.

**pandas**: Provides a DataFrame structure to store data in memory and work with
it easily and efficiently.

**scikit-learn**: The essential Machine Learning package for a variaty of
supervised learning models, in Python.

**tensorflow**: The essential Machine Learning package for deep learning, in
Python.

**matplotlib**: Basic plotting library in Python; most other Python plotting
libraries are built on top of it.

### Models
Regarding the supervised learning models, we are using:
* [Gaussian Naive Bayes](https://scikit-
learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
* [Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
* [DecisionTree](https://scikit-
learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
* [XGBoost](https://xgboost.readthedocs.io/en/latest/)
* [Neural Networks](https://keras.io/guides/sequential_model/)
* [Deep Neural Networks](https://keras.io/guides/sequential_model/)

```{.python .input  n=141}
# Primary libraries
from time import time
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
# Neural Networks
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
# Measures
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder  
```

## The problem domain
[go back to the top](#Table-of-contents)

The first step to any data analysis project is to define the question or problem
we're looking to solve, and to define a measure (or set of measures) for our
success at solving that task. The data analysis checklist has us answer a
handful of questions to accomplish that, so let's work through those questions.

#### Did you specify the type of data analytic question (e.g. exploration,
association causality) before touching the data?

> We are trying to design a predictive model capable of accurately predicting if
the home team will either win, lose or draw, i.e., predict the outcome of
football matche based on a set of measurements, including player ratings, team
ratings, team average stats(possession, corners, shoots), team style(pressing,
possession, defending, counter attacking, speed of play, ..) and team match
history(previous games).

#### Did you define the metric for success before beginning?

> Let's do that now. Since we're performing classification, we can use
[accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision) ÔÇö the fraction
of correctly classified matches ÔÇö to quantify how well our model is performing.
Knowing that most bookkeepers predict matches with an accuracy of 50%, we will
try to match or beat that value. We will also use a confusion matrix, and
analyse the precision, recall and f1-score.

#### Did you consider whether the question could be answered with the available
data?",

> The data provided has information about more than 25k matches across multiple
leagues. Even though the usability isn't great, after some processing and
cleansing of the data, we will be able to predict matches with great confidence.
To answer the question, yes, we have more than enough data to analyse football
matches..

## Step 1: Data analysis
[go back to the top](#Table-of-contents)

The first step we have, is to look at the data, and after extracting, analyse
it. We know that most datasets can contain minor issues, so we have to search
for  possible null or not defined values, and if so how do we proceed? Do we
remove an entire row of a Dataframe? Maybe we just need to purify and substitute
it's value? This analysis is done below.

Before analysing the data, we need to first extract it. For that we use multiple
methods to have a cleaner code

### Extracting data from the database

```{.python .input  n=142}
with sqlite3.connect("../dataset/database.sqlite") as con:
    matches = pd.read_sql_query("SELECT * from Match", con)
    team_attributes = pd.read_sql_query("SELECT distinct * from Team_Attributes",con)
    player = pd.read_sql_query("SELECT * from Player",con)
    player_attributes = pd.read_sql_query("SELECT * from Player_Attributes",con)
```

### Matches
We start by cleaning the match data and defining some methods for the data
extraction and the labels

```{.python .input  n=143}
''' Derives a label for a given match. '''
def get_match_outcome(match):
    
    #Define variables
    home_goals = match['home_team_goal']
    away_goals = match['away_team_goal']
     
    outcome = pd.DataFrame()
    outcome.loc[0,'match_api_id'] = match['match_api_id'] 

    #Identify match outcome  
    if home_goals > away_goals:
        outcome.loc[0,'outcome'] = "Win"
    if home_goals == away_goals:
        outcome.loc[0,'outcome'] = "Draw"
    if home_goals < away_goals:
        outcome.loc[0,'outcome'] = "Defeat"

    #Return outcome        
    return outcome.loc[0]


''' Get the last x matches of a given team. '''
def get_last_matches(matches, date, team, x = 10):
    #Filter team matches from matches
    team_matches = matches[(matches['home_team_api_id'] == team) | (matches['away_team_api_id'] == team)]
                           
    #Filter x last matches from team matches
    last_matches = team_matches[team_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:x,:]
    
    #Return last matches
    return last_matches
    
    
''' Get the last team stats of a given team. '''
def get_last_team_stats(team_id, date, team_stats):
    #Filter team stats
    all_team_stats = teams_stats[teams_stats['team_api_id'] == team_id]
           
    #Filter last stats from team
    last_team_stats = all_team_stats[all_team_stats.date < date].sort_values(by='date', ascending=False)
    if last_team_stats.empty:
        last_team_stats = all_team_stats[all_team_stats.date > date].sort_values(by='date', ascending=True)

    #Return last matches
    return last_team_stats.iloc[0:1,:]
    
    
''' Get the last x matches of two given teams. '''
def get_last_matches_against_eachother(matches, date, home_team, away_team, x = 10):
    #Find matches of both teams
    home_matches = matches[(matches['home_team_api_id'] == home_team) & (matches['away_team_api_id'] == away_team)]    
    away_matches = matches[(matches['home_team_api_id'] == away_team) & (matches['away_team_api_id'] == home_team)]  
    total_matches = pd.concat([home_matches, away_matches])
    
    #Get last x matches
    try:    
        last_matches = total_matches[total_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:x,:]
    except:
        last_matches = total_matches[total_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:total_matches.shape[0],:]
        
        #Check for error in data
        if(last_matches.shape[0] > x):
            print("Error in obtaining matches")
            
    #Return data
    return last_matches


''' Get the goals[home & away] of a specfic team from a set of matches. '''
def get_goals(matches, team):
    home_goals = int(matches.home_team_goal[matches.home_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.away_team_api_id == team].sum())

    total_goals = home_goals + away_goals
    
    return total_goals


''' Get the goals[home & away] conceided of a specfic team from a set of matches. '''
def get_goals_conceided(matches, team):
    home_goals = int(matches.home_team_goal[matches.away_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.home_team_api_id == team].sum())

    total_goals = home_goals + away_goals

    return total_goals


''' Get the number of wins of a specfic team from a set of matches. '''
def get_wins(matches, team):
    #Find home and away wins
    home_wins = int(matches.home_team_goal[(matches.home_team_api_id == team) & (matches.home_team_goal > matches.away_team_goal)].count())
    away_wins = int(matches.away_team_goal[(matches.away_team_api_id == team) & (matches.away_team_goal > matches.home_team_goal)].count())

    total_wins = home_wins + away_wins

    return total_wins 


''' Create match specific features for a given match. '''
def get_match_features(match, matches, teams_stats, x = 10):
    #Define variables
    date = match.date
    home_team = match.home_team_api_id
    away_team = match.away_team_api_id
    
     # Gets home and away team_stats
    home_team_stats = get_last_team_stats(home_team, date, teams_stats);
    away_team_stats = get_last_team_stats(away_team, date, teams_stats);
    
    #Get last x matches of home and away team
    matches_home_team = get_last_matches(matches, date, home_team, x = 5)
    matches_away_team = get_last_matches(matches, date, away_team, x = 5)
    
    #Get last x matches of both teams against each other
    last_matches_against = get_last_matches_against_eachother(matches, date, home_team, away_team, x = 3)
    
    #Create goal variables
    home_goals = get_goals(matches_home_team, home_team)
    away_goals = get_goals(matches_away_team, away_team)
    home_goals_conceided = get_goals_conceided(matches_home_team, home_team)
    away_goals_conceided = get_goals_conceided(matches_away_team, away_team)
    
    #Define result data frame
    result = pd.DataFrame()
    
    #Define ID features
    result.loc[0, 'match_api_id'] = match.match_api_id
    result.loc[0, 'league_id'] = match.league_id
    
    #Create match features and team stats
    if(not home_team_stats.empty):
        result.loc[0, 'home_team_buildUpPlaySpeed'] = home_team_stats['buildUpPlaySpeed'].values[0]
        result.loc[0, 'home_team_buildUpPlayPassing'] = home_team_stats['buildUpPlayPassing'].values[0]
        result.loc[0, 'home_team_chanceCreationPassing'] = home_team_stats['chanceCreationPassing'].values[0]
        result.loc[0, 'home_team_chanceCreationCrossing'] = home_team_stats['chanceCreationCrossing'].values[0]
        result.loc[0, 'home_team_chanceCreationShooting'] = home_team_stats['chanceCreationShooting'].values[0]
        result.loc[0, 'home_team_defencePressure'] = home_team_stats['defencePressure'].values[0]
        result.loc[0, 'home_team_defenceAggression'] = home_team_stats['defenceAggression'].values[0]
        result.loc[0, 'home_team_defenceTeamWidth'] = home_team_stats['defenceTeamWidth'].values[0]
        result.loc[0, 'home_team_avg_shots'] = home_team_stats['avg_shots'].values[0]
        result.loc[0, 'home_team_avg_corners'] = home_team_stats['avg_corners'].values[0]
        result.loc[0, 'home_team_avg_crosses'] = away_team_stats['avg_crosses'].values[0]
    
    if(not away_team_stats.empty):
        result.loc[0, 'away_team_buildUpPlaySpeed'] = away_team_stats['buildUpPlaySpeed'].values[0]
        result.loc[0, 'away_team_buildUpPlayPassing'] = away_team_stats['buildUpPlayPassing'].values[0]
        result.loc[0, 'away_team_chanceCreationPassing'] = away_team_stats['chanceCreationPassing'].values[0]
        result.loc[0, 'away_team_chanceCreationCrossing'] = away_team_stats['chanceCreationCrossing'].values[0]
        result.loc[0, 'away_team_chanceCreationShooting'] = away_team_stats['chanceCreationShooting'].values[0]
        result.loc[0, 'away_team_defencePressure'] = away_team_stats['defencePressure'].values[0]
        result.loc[0, 'away_team_defenceAggression'] = away_team_stats['defenceAggression'].values[0]
        result.loc[0, 'away_team_defenceTeamWidth'] = away_team_stats['defenceTeamWidth'].values[0]
        result.loc[0, 'away_team_avg_shots'] = away_team_stats['avg_shots'].values[0]
        result.loc[0, 'away_team_avg_corners'] = away_team_stats['avg_corners'].values[0]
        result.loc[0, 'away_team_avg_crosses'] = away_team_stats['avg_crosses'].values[0]
    
    result.loc[0, 'home_team_goals_difference'] = home_goals - home_goals_conceided
    result.loc[0, 'away_team_goals_difference'] = away_goals - away_goals_conceided
    result.loc[0, 'games_won_home_team'] = get_wins(matches_home_team, home_team) 
    result.loc[0, 'games_won_away_team'] = get_wins(matches_away_team, away_team)
    result.loc[0, 'games_against_won'] = get_wins(last_matches_against, home_team)
    result.loc[0, 'games_against_lost'] = get_wins(last_matches_against, away_team)
    result.loc[0, 'B365H'] = match.B365H
    result.loc[0, 'B365D'] = match.B365D
    result.loc[0, 'B365A'] = match.B365A
    
    #Return match features
    return result.loc[0]

''' Create and aggregate features and labels for all matches. '''
def get_features(matches, teams_stats, fifa, x = 10, get_overall = False):  
    #Get fifa stats features
    fifa_stats = get_overall_fifa_rankings(fifa, get_overall)
    
    #Get match features for all matches
    match_stats = matches.apply(lambda i: get_match_features(i, matches, teams_stats, x = 10), axis = 1)
    
    #Create dummies for league ID feature
    dummies = pd.get_dummies(match_stats['league_id']).rename(columns = lambda x: 'League_' + str(x))
    match_stats = pd.concat([match_stats, dummies], axis = 1)
    match_stats.drop(['league_id'], inplace = True, axis = 1)
    
    #Create match outcomes
    outcomes = matches.apply(get_match_outcome, axis = 1)

    #Merges features and outcomes into one frame
    features = pd.merge(match_stats, fifa_stats, on = 'match_api_id', how = 'left')
    features = pd.merge(features, outcomes, on = 'match_api_id', how = 'left')
    
    #Drop NA values
    features.dropna(inplace = True)
    
    #Return preprocessed data
    return features


def get_overall_fifa_rankings(fifa, get_overall = False):
    ''' Get overall fifa rankings from fifa data. '''
      
    temp_data = fifa
    
    #Check if only overall player stats are desired
    if get_overall == True:
        
        #Get overall stats
        data = temp_data.loc[:,(fifa.columns.str.contains('overall_rating'))]
        data.loc[:,'match_api_id'] = temp_data.loc[:,'match_api_id']
    else:
        
        #Get all stats except for stat date
        cols = fifa.loc[:,(fifa.columns.str.contains('date_stat'))]
        temp_data = fifa.drop(cols.columns, axis = 1)        
        data = temp_data
    
    #Return data
    return data
```

```{.python .input  n=144}
viable_matches = matches
viable_matches.describe()
```

```{.json .output n=144}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>country_id</th>\n      <th>league_id</th>\n      <th>stage</th>\n      <th>match_api_id</th>\n      <th>home_team_api_id</th>\n      <th>away_team_api_id</th>\n      <th>home_team_goal</th>\n      <th>away_team_goal</th>\n      <th>home_player_X1</th>\n      <th>...</th>\n      <th>SJA</th>\n      <th>VCH</th>\n      <th>VCD</th>\n      <th>VCA</th>\n      <th>GBH</th>\n      <th>GBD</th>\n      <th>GBA</th>\n      <th>BSH</th>\n      <th>BSD</th>\n      <th>BSA</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>25979.000000</td>\n      <td>25979.000000</td>\n      <td>25979.000000</td>\n      <td>25979.000000</td>\n      <td>2.597900e+04</td>\n      <td>25979.000000</td>\n      <td>25979.000000</td>\n      <td>25979.000000</td>\n      <td>25979.000000</td>\n      <td>24158.000000</td>\n      <td>...</td>\n      <td>17097.000000</td>\n      <td>22568.000000</td>\n      <td>22568.000000</td>\n      <td>22568.000000</td>\n      <td>14162.000000</td>\n      <td>14162.000000</td>\n      <td>14162.000000</td>\n      <td>14161.000000</td>\n      <td>14161.000000</td>\n      <td>14161.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>12990.000000</td>\n      <td>11738.630317</td>\n      <td>11738.630317</td>\n      <td>18.242773</td>\n      <td>1.195429e+06</td>\n      <td>9984.371993</td>\n      <td>9984.475115</td>\n      <td>1.544594</td>\n      <td>1.160938</td>\n      <td>0.999586</td>\n      <td>...</td>\n      <td>4.622343</td>\n      <td>2.668107</td>\n      <td>3.899048</td>\n      <td>4.840281</td>\n      <td>2.498764</td>\n      <td>3.648189</td>\n      <td>4.353097</td>\n      <td>2.497894</td>\n      <td>3.660742</td>\n      <td>4.405663</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>7499.635658</td>\n      <td>7553.936759</td>\n      <td>7553.936759</td>\n      <td>10.407354</td>\n      <td>4.946279e+05</td>\n      <td>14087.453758</td>\n      <td>14087.445135</td>\n      <td>1.297158</td>\n      <td>1.142110</td>\n      <td>0.022284</td>\n      <td>...</td>\n      <td>3.632164</td>\n      <td>1.928753</td>\n      <td>1.248221</td>\n      <td>4.318338</td>\n      <td>1.489299</td>\n      <td>0.867440</td>\n      <td>3.010189</td>\n      <td>1.507793</td>\n      <td>0.868272</td>\n      <td>3.189814</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>4.831290e+05</td>\n      <td>1601.000000</td>\n      <td>1601.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>...</td>\n      <td>1.100000</td>\n      <td>1.030000</td>\n      <td>1.620000</td>\n      <td>1.080000</td>\n      <td>1.050000</td>\n      <td>1.450000</td>\n      <td>1.120000</td>\n      <td>1.040000</td>\n      <td>1.330000</td>\n      <td>1.120000</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>6495.500000</td>\n      <td>4769.000000</td>\n      <td>4769.000000</td>\n      <td>9.000000</td>\n      <td>7.684365e+05</td>\n      <td>8475.000000</td>\n      <td>8475.000000</td>\n      <td>1.000000</td>\n      <td>0.000000</td>\n      <td>1.000000</td>\n      <td>...</td>\n      <td>2.500000</td>\n      <td>1.700000</td>\n      <td>3.300000</td>\n      <td>2.550000</td>\n      <td>1.670000</td>\n      <td>3.200000</td>\n      <td>2.500000</td>\n      <td>1.670000</td>\n      <td>3.250000</td>\n      <td>2.500000</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>12990.000000</td>\n      <td>10257.000000</td>\n      <td>10257.000000</td>\n      <td>18.000000</td>\n      <td>1.147511e+06</td>\n      <td>8697.000000</td>\n      <td>8697.000000</td>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>...</td>\n      <td>3.500000</td>\n      <td>2.150000</td>\n      <td>3.500000</td>\n      <td>3.500000</td>\n      <td>2.100000</td>\n      <td>3.300000</td>\n      <td>3.400000</td>\n      <td>2.100000</td>\n      <td>3.400000</td>\n      <td>3.400000</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>19484.500000</td>\n      <td>17642.000000</td>\n      <td>17642.000000</td>\n      <td>27.000000</td>\n      <td>1.709852e+06</td>\n      <td>9925.000000</td>\n      <td>9925.000000</td>\n      <td>2.000000</td>\n      <td>2.000000</td>\n      <td>1.000000</td>\n      <td>...</td>\n      <td>5.250000</td>\n      <td>2.800000</td>\n      <td>4.000000</td>\n      <td>5.400000</td>\n      <td>2.650000</td>\n      <td>3.750000</td>\n      <td>5.000000</td>\n      <td>2.620000</td>\n      <td>3.750000</td>\n      <td>5.000000</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>25979.000000</td>\n      <td>24558.000000</td>\n      <td>24558.000000</td>\n      <td>38.000000</td>\n      <td>2.216672e+06</td>\n      <td>274581.000000</td>\n      <td>274581.000000</td>\n      <td>10.000000</td>\n      <td>9.000000</td>\n      <td>2.000000</td>\n      <td>...</td>\n      <td>41.000000</td>\n      <td>36.000000</td>\n      <td>26.000000</td>\n      <td>67.000000</td>\n      <td>21.000000</td>\n      <td>11.000000</td>\n      <td>34.000000</td>\n      <td>17.000000</td>\n      <td>13.000000</td>\n      <td>34.000000</td>\n    </tr>\n  </tbody>\n</table>\n<p>8 rows \u00d7 105 columns</p>\n</div>",
   "text/plain": "                 id    country_id     league_id         stage  match_api_id  \\\ncount  25979.000000  25979.000000  25979.000000  25979.000000  2.597900e+04   \nmean   12990.000000  11738.630317  11738.630317     18.242773  1.195429e+06   \nstd     7499.635658   7553.936759   7553.936759     10.407354  4.946279e+05   \nmin        1.000000      1.000000      1.000000      1.000000  4.831290e+05   \n25%     6495.500000   4769.000000   4769.000000      9.000000  7.684365e+05   \n50%    12990.000000  10257.000000  10257.000000     18.000000  1.147511e+06   \n75%    19484.500000  17642.000000  17642.000000     27.000000  1.709852e+06   \nmax    25979.000000  24558.000000  24558.000000     38.000000  2.216672e+06   \n\n       home_team_api_id  away_team_api_id  home_team_goal  away_team_goal  \\\ncount      25979.000000      25979.000000    25979.000000    25979.000000   \nmean        9984.371993       9984.475115        1.544594        1.160938   \nstd        14087.453758      14087.445135        1.297158        1.142110   \nmin         1601.000000       1601.000000        0.000000        0.000000   \n25%         8475.000000       8475.000000        1.000000        0.000000   \n50%         8697.000000       8697.000000        1.000000        1.000000   \n75%         9925.000000       9925.000000        2.000000        2.000000   \nmax       274581.000000     274581.000000       10.000000        9.000000   \n\n       home_player_X1  ...           SJA           VCH           VCD  \\\ncount    24158.000000  ...  17097.000000  22568.000000  22568.000000   \nmean         0.999586  ...      4.622343      2.668107      3.899048   \nstd          0.022284  ...      3.632164      1.928753      1.248221   \nmin          0.000000  ...      1.100000      1.030000      1.620000   \n25%          1.000000  ...      2.500000      1.700000      3.300000   \n50%          1.000000  ...      3.500000      2.150000      3.500000   \n75%          1.000000  ...      5.250000      2.800000      4.000000   \nmax          2.000000  ...     41.000000     36.000000     26.000000   \n\n                VCA           GBH           GBD           GBA           BSH  \\\ncount  22568.000000  14162.000000  14162.000000  14162.000000  14161.000000   \nmean       4.840281      2.498764      3.648189      4.353097      2.497894   \nstd        4.318338      1.489299      0.867440      3.010189      1.507793   \nmin        1.080000      1.050000      1.450000      1.120000      1.040000   \n25%        2.550000      1.670000      3.200000      2.500000      1.670000   \n50%        3.500000      2.100000      3.300000      3.400000      2.100000   \n75%        5.400000      2.650000      3.750000      5.000000      2.620000   \nmax       67.000000     21.000000     11.000000     34.000000     17.000000   \n\n                BSD           BSA  \ncount  14161.000000  14161.000000  \nmean       3.660742      4.405663  \nstd        0.868272      3.189814  \nmin        1.330000      1.120000  \n25%        3.250000      2.500000  \n50%        3.400000      3.400000  \n75%        3.750000      5.000000  \nmax       13.000000     34.000000  \n\n[8 rows x 105 columns]"
  },
  "execution_count": 144,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Looking at the match data we can see that most columns have 25979 values. This
means we are analysing this number of matches from the database. We can start by
looking at the bookkeeper data. We can see that the number of bookkepper match
data is different for each bookkeper. We start by selecting the bookeeper with
the most predictions data available.

```{.python .input  n=145}
viable_matches = matches.sample(n=5000)
b365 = viable_matches.dropna(subset=['B365H', 'B365D', 'B365A'],inplace=False)
b365.drop(['BWH', 'BWD', 'BWA', 
          'IWH', 'IWD', 'IWA',  
          'LBH', 'LBD', 'LBA',
          'PSH', 'PSD', 'PSA',
          'WHH', 'WHD', 'WHA',
          'SJH', 'SJD', 'SJA',
          'VCH', 'VCD', 'VCA', 
          'GBH', 'GBD', 'GBA', 
          'BSH', 'BSD', 'BSA'], inplace = True, axis = 1)

bw = viable_matches.dropna(subset=['BWH', 'BWD', 'BWA'],inplace=False)
bw.drop(['B365H', 'B365D', 'B365A', 
          'IWH', 'IWD', 'IWA',  
          'LBH', 'LBD', 'LBA',
          'PSH', 'PSD', 'PSA',
          'WHH', 'WHD', 'WHA',
          'SJH', 'SJD', 'SJA',
          'VCH', 'VCD', 'VCA', 
          'GBH', 'GBD', 'GBA', 
          'BSH', 'BSD', 'BSA'], inplace=True, axis = 1)

iw = viable_matches.dropna(subset=['IWH', 'IWD', 'IWA'],inplace=False)
iw.drop(['B365H', 'B365D', 'B365A', 
          'BWH', 'BWD', 'BWA',  
          'LBH', 'LBD', 'LBA',
          'PSH', 'PSD', 'PSA',
          'WHH', 'WHD', 'WHA',
          'SJH', 'SJD', 'SJA',
          'VCH', 'VCD', 'VCA', 
          'GBH', 'GBD', 'GBA', 
          'BSH', 'BSD', 'BSA'], inplace=True, axis = 1)

lb = viable_matches.dropna(subset=['LBH', 'LBD', 'LBA'],inplace=False)
lb.drop(['B365H', 'B365D', 'B365A', 
          'BWH', 'BWD', 'BWA',  
          'IWH', 'IWD', 'IWA',
          'PSH', 'PSD', 'PSA',
          'WHH', 'WHD', 'WHA',
          'SJH', 'SJD', 'SJA',
          'VCH', 'VCD', 'VCA', 
          'GBH', 'GBD', 'GBA', 
          'BSH', 'BSD', 'BSA'], inplace=True, axis = 1)

ps = viable_matches.dropna(subset=['PSH', 'PSD', 'PSA'],inplace=False)
ps.drop(['B365H', 'B365D', 'B365A', 
          'BWH', 'BWD', 'BWA',  
          'IWH', 'IWD', 'IWA',
          'LBH', 'LBD', 'LBA',
          'WHH', 'WHD', 'WHA',
          'SJH', 'SJD', 'SJA',
          'VCH', 'VCD', 'VCA', 
          'GBH', 'GBD', 'GBA', 
          'BSH', 'BSD', 'BSA'], inplace=True, axis = 1)

wh = viable_matches.dropna(subset=['WHH', 'WHD', 'WHA'],inplace=False)
wh.drop(['B365H', 'B365D', 'B365A', 
          'BWH', 'BWD', 'BWA',  
          'IWH', 'IWD', 'IWA',
          'LBH', 'LBD', 'LBA',
          'PSH', 'PSD', 'PSA',
          'SJH', 'SJD', 'SJA',
          'VCH', 'VCD', 'VCA', 
          'GBH', 'GBD', 'GBA', 
          'BSH', 'BSD', 'BSA'], inplace=True, axis = 1)

sj = viable_matches.dropna(subset=['SJH', 'SJD', 'SJA'],inplace=False)
sj.drop(['B365H', 'B365D', 'B365A', 
          'BWH', 'BWD', 'BWA',  
          'IWH', 'IWD', 'IWA',
          'LBH', 'LBD', 'LBA',
          'PSH', 'PSD', 'PSA',
          'WHH', 'WHD', 'WHA',
          'VCH', 'VCD', 'VCA', 
          'GBH', 'GBD', 'GBA', 
          'BSH', 'BSD', 'BSA'], inplace=True, axis = 1)

vc = viable_matches.dropna(subset=['VCH', 'VCD', 'VCA'],inplace=False)
vc.drop(['B365H', 'B365D', 'B365A', 
          'BWH', 'BWD', 'BWA',  
          'IWH', 'IWD', 'IWA',
          'LBH', 'LBD', 'LBA',
          'PSH', 'PSD', 'PSA',
          'WHH', 'WHD', 'WHA',
          'SJH', 'SJD', 'SJA', 
          'GBH', 'GBD', 'GBA', 
          'BSH', 'BSD', 'BSA'], inplace=True, axis = 1)

gb = viable_matches.dropna(subset=['GBH', 'GBD', 'GBA'],inplace=False)
gb.drop(['B365H', 'B365D', 'B365A', 
          'BWH', 'BWD', 'BWA',  
          'IWH', 'IWD', 'IWA',
          'LBH', 'LBD', 'LBA',
          'PSH', 'PSD', 'PSA',
          'WHH', 'WHD', 'WHA',
          'SJH', 'SJD', 'SJA', 
          'VCH', 'VCD', 'VCA', 
          'BSH', 'BSD', 'BSA'], inplace=True, axis = 1)

bs = viable_matches.dropna(subset=['BSH', 'BSD', 'BSA'],inplace=False)
bs.drop(['B365H', 'B365D', 'B365A', 
          'BWH', 'BWD', 'BWA',  
          'IWH', 'IWD', 'IWA',
          'LBH', 'LBD', 'LBA',
          'PSH', 'PSD', 'PSA',
          'WHH', 'WHD', 'WHA',
          'SJH', 'SJD', 'SJA', 
          'VCH', 'VCD', 'VCA', 
          'GBH', 'GBD', 'GBA'], inplace=True, axis = 1)

lis = [b365, bw, iw, lb, ps, wh, sj, vc, gb, bs]

viable_matches = max(lis, key = lambda datframe: datframe.shape[0])
viable_matches.describe()
```

```{.json .output n=145}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "C:\\Python\\Python38\\lib\\site-packages\\pandas\\core\\frame.py:3990: SettingWithCopyWarning: \nA value is trying to be set on a copy of a slice from a DataFrame\n\nSee the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n  return super().drop(\n"
 },
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>country_id</th>\n      <th>league_id</th>\n      <th>stage</th>\n      <th>match_api_id</th>\n      <th>home_team_api_id</th>\n      <th>away_team_api_id</th>\n      <th>home_team_goal</th>\n      <th>away_team_goal</th>\n      <th>home_player_X1</th>\n      <th>...</th>\n      <th>away_player_5</th>\n      <th>away_player_6</th>\n      <th>away_player_7</th>\n      <th>away_player_8</th>\n      <th>away_player_9</th>\n      <th>away_player_10</th>\n      <th>away_player_11</th>\n      <th>B365H</th>\n      <th>B365D</th>\n      <th>B365A</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>4320.000000</td>\n      <td>4320.000000</td>\n      <td>4320.000000</td>\n      <td>4320.000000</td>\n      <td>4.320000e+03</td>\n      <td>4320.000000</td>\n      <td>4320.000000</td>\n      <td>4320.000000</td>\n      <td>4320.000000</td>\n      <td>4216.000000</td>\n      <td>...</td>\n      <td>4194.000000</td>\n      <td>4188.000000</td>\n      <td>4193.000000</td>\n      <td>4175.000000</td>\n      <td>4193.000000</td>\n      <td>4185.000000</td>\n      <td>4180.000000</td>\n      <td>4320.000000</td>\n      <td>4320.000000</td>\n      <td>4320.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>11814.629398</td>\n      <td>10519.550463</td>\n      <td>10519.550463</td>\n      <td>18.659722</td>\n      <td>1.184003e+06</td>\n      <td>10052.549074</td>\n      <td>10127.751157</td>\n      <td>1.541667</td>\n      <td>1.147454</td>\n      <td>0.999526</td>\n      <td>...</td>\n      <td>109347.383643</td>\n      <td>96893.740688</td>\n      <td>92917.334844</td>\n      <td>106707.578443</td>\n      <td>109751.488195</td>\n      <td>105582.187575</td>\n      <td>103511.687560</td>\n      <td>2.585648</td>\n      <td>3.820461</td>\n      <td>4.635725</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>7276.383156</td>\n      <td>7267.357816</td>\n      <td>7267.357816</td>\n      <td>10.576109</td>\n      <td>4.978897e+05</td>\n      <td>13798.515929</td>\n      <td>13839.733497</td>\n      <td>1.301903</td>\n      <td>1.118067</td>\n      <td>0.030802</td>\n      <td>...</td>\n      <td>112440.605262</td>\n      <td>107353.223742</td>\n      <td>102685.156581</td>\n      <td>115664.276539</td>\n      <td>115409.084952</td>\n      <td>114181.111654</td>\n      <td>110172.627268</td>\n      <td>1.640789</td>\n      <td>1.127839</td>\n      <td>3.694938</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>2.000000</td>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>4.831300e+05</td>\n      <td>1773.000000</td>\n      <td>1773.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>...</td>\n      <td>2790.000000</td>\n      <td>2802.000000</td>\n      <td>2770.000000</td>\n      <td>2802.000000</td>\n      <td>2770.000000</td>\n      <td>2802.000000</td>\n      <td>2862.000000</td>\n      <td>1.040000</td>\n      <td>1.730000</td>\n      <td>1.170000</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>5609.500000</td>\n      <td>4769.000000</td>\n      <td>4769.000000</td>\n      <td>10.000000</td>\n      <td>7.049020e+05</td>\n      <td>8529.000000</td>\n      <td>8529.000000</td>\n      <td>1.000000</td>\n      <td>0.000000</td>\n      <td>1.000000</td>\n      <td>...</td>\n      <td>33338.000000</td>\n      <td>30920.000000</td>\n      <td>30872.000000</td>\n      <td>32575.000000</td>\n      <td>32924.000000</td>\n      <td>31921.000000</td>\n      <td>32695.750000</td>\n      <td>1.670000</td>\n      <td>3.300000</td>\n      <td>2.500000</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>11237.500000</td>\n      <td>10257.000000</td>\n      <td>10257.000000</td>\n      <td>18.000000</td>\n      <td>1.083005e+06</td>\n      <td>8721.000000</td>\n      <td>8721.000000</td>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>...</td>\n      <td>45276.000000</td>\n      <td>40166.000000</td>\n      <td>40444.000000</td>\n      <td>42321.000000</td>\n      <td>42706.000000</td>\n      <td>41542.000000</td>\n      <td>41387.500000</td>\n      <td>2.100000</td>\n      <td>3.400000</td>\n      <td>3.500000</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>18719.000000</td>\n      <td>17642.000000</td>\n      <td>17642.000000</td>\n      <td>28.000000</td>\n      <td>1.574226e+06</td>\n      <td>9906.000000</td>\n      <td>9905.000000</td>\n      <td>2.000000</td>\n      <td>2.000000</td>\n      <td>1.000000</td>\n      <td>...</td>\n      <td>161414.000000</td>\n      <td>149962.000000</td>\n      <td>134217.000000</td>\n      <td>156551.000000</td>\n      <td>164105.000000</td>\n      <td>161291.000000</td>\n      <td>161420.000000</td>\n      <td>2.800000</td>\n      <td>3.800000</td>\n      <td>5.250000</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>24557.000000</td>\n      <td>21518.000000</td>\n      <td>21518.000000</td>\n      <td>38.000000</td>\n      <td>2.216672e+06</td>\n      <td>274581.000000</td>\n      <td>274581.000000</td>\n      <td>10.000000</td>\n      <td>8.000000</td>\n      <td>2.000000</td>\n      <td>...</td>\n      <td>721133.000000</td>\n      <td>680905.000000</td>\n      <td>701154.000000</td>\n      <td>704523.000000</td>\n      <td>684978.000000</td>\n      <td>722766.000000</td>\n      <td>725718.000000</td>\n      <td>15.000000</td>\n      <td>17.000000</td>\n      <td>41.000000</td>\n    </tr>\n  </tbody>\n</table>\n<p>8 rows \u00d7 78 columns</p>\n</div>",
   "text/plain": "                 id    country_id     league_id        stage  match_api_id  \\\ncount   4320.000000   4320.000000   4320.000000  4320.000000  4.320000e+03   \nmean   11814.629398  10519.550463  10519.550463    18.659722  1.184003e+06   \nstd     7276.383156   7267.357816   7267.357816    10.576109  4.978897e+05   \nmin        2.000000      1.000000      1.000000     1.000000  4.831300e+05   \n25%     5609.500000   4769.000000   4769.000000    10.000000  7.049020e+05   \n50%    11237.500000  10257.000000  10257.000000    18.000000  1.083005e+06   \n75%    18719.000000  17642.000000  17642.000000    28.000000  1.574226e+06   \nmax    24557.000000  21518.000000  21518.000000    38.000000  2.216672e+06   \n\n       home_team_api_id  away_team_api_id  home_team_goal  away_team_goal  \\\ncount       4320.000000       4320.000000     4320.000000     4320.000000   \nmean       10052.549074      10127.751157        1.541667        1.147454   \nstd        13798.515929      13839.733497        1.301903        1.118067   \nmin         1773.000000       1773.000000        0.000000        0.000000   \n25%         8529.000000       8529.000000        1.000000        0.000000   \n50%         8721.000000       8721.000000        1.000000        1.000000   \n75%         9906.000000       9905.000000        2.000000        2.000000   \nmax       274581.000000     274581.000000       10.000000        8.000000   \n\n       home_player_X1  ...  away_player_5  away_player_6  away_player_7  \\\ncount     4216.000000  ...    4194.000000    4188.000000    4193.000000   \nmean         0.999526  ...  109347.383643   96893.740688   92917.334844   \nstd          0.030802  ...  112440.605262  107353.223742  102685.156581   \nmin          0.000000  ...    2790.000000    2802.000000    2770.000000   \n25%          1.000000  ...   33338.000000   30920.000000   30872.000000   \n50%          1.000000  ...   45276.000000   40166.000000   40444.000000   \n75%          1.000000  ...  161414.000000  149962.000000  134217.000000   \nmax          2.000000  ...  721133.000000  680905.000000  701154.000000   \n\n       away_player_8  away_player_9  away_player_10  away_player_11  \\\ncount    4175.000000    4193.000000     4185.000000     4180.000000   \nmean   106707.578443  109751.488195   105582.187575   103511.687560   \nstd    115664.276539  115409.084952   114181.111654   110172.627268   \nmin      2802.000000    2770.000000     2802.000000     2862.000000   \n25%     32575.000000   32924.000000    31921.000000    32695.750000   \n50%     42321.000000   42706.000000    41542.000000    41387.500000   \n75%    156551.000000  164105.000000   161291.000000   161420.000000   \nmax    704523.000000  684978.000000   722766.000000   725718.000000   \n\n             B365H        B365D        B365A  \ncount  4320.000000  4320.000000  4320.000000  \nmean      2.585648     3.820461     4.635725  \nstd       1.640789     1.127839     3.694938  \nmin       1.040000     1.730000     1.170000  \n25%       1.670000     3.300000     2.500000  \n50%       2.100000     3.400000     3.500000  \n75%       2.800000     3.800000     5.250000  \nmax      15.000000    17.000000    41.000000  \n\n[8 rows x 78 columns]"
  },
  "execution_count": 145,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Analysing the description of the dataframe, we can see that the bookkeper
regarding Bet 365 has the most available information and, has such, we will
decide to selected it as a feature input for our models.

We also need to consider that some of these matches may not be on the team
attributes that we will clean after this. In that case, we need to remove any
matches that does not contain any team stats information, since **mean
imputation** would't work in these case.

We also need to remove some rows that do not contain any information about the
position of the players for some matches.

```{.python .input  n=146}
teams_stats = team_attributes
viable_matches = viable_matches.dropna(inplace=False)

home_teams = viable_matches['home_team_api_id'].isin(teams_stats['team_api_id'].tolist())
away_teams = viable_matches['away_team_api_id'].isin(teams_stats['team_api_id'].tolist())
viable_matches = viable_matches[home_teams & away_teams]

viable_matches.describe()
```

```{.json .output n=146}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>country_id</th>\n      <th>league_id</th>\n      <th>stage</th>\n      <th>match_api_id</th>\n      <th>home_team_api_id</th>\n      <th>away_team_api_id</th>\n      <th>home_team_goal</th>\n      <th>away_team_goal</th>\n      <th>home_player_X1</th>\n      <th>...</th>\n      <th>away_player_5</th>\n      <th>away_player_6</th>\n      <th>away_player_7</th>\n      <th>away_player_8</th>\n      <th>away_player_9</th>\n      <th>away_player_10</th>\n      <th>away_player_11</th>\n      <th>B365H</th>\n      <th>B365D</th>\n      <th>B365A</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2.505000e+03</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.0</td>\n      <td>...</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>11050.111377</td>\n      <td>9471.790419</td>\n      <td>9471.790419</td>\n      <td>19.441118</td>\n      <td>1.273188e+06</td>\n      <td>9326.698204</td>\n      <td>9268.572854</td>\n      <td>1.543313</td>\n      <td>1.144112</td>\n      <td>1.0</td>\n      <td>...</td>\n      <td>103986.502994</td>\n      <td>95100.075050</td>\n      <td>91516.336527</td>\n      <td>105081.580439</td>\n      <td>105003.107784</td>\n      <td>102028.245509</td>\n      <td>96260.831537</td>\n      <td>2.558671</td>\n      <td>3.857697</td>\n      <td>4.738930</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>6971.115807</td>\n      <td>6923.153417</td>\n      <td>6923.153417</td>\n      <td>10.744260</td>\n      <td>4.837662e+05</td>\n      <td>6368.675817</td>\n      <td>4958.688585</td>\n      <td>1.288241</td>\n      <td>1.113556</td>\n      <td>0.0</td>\n      <td>...</td>\n      <td>112696.240000</td>\n      <td>106123.109905</td>\n      <td>103938.792275</td>\n      <td>114656.410740</td>\n      <td>115588.861261</td>\n      <td>113190.241096</td>\n      <td>108851.544990</td>\n      <td>1.618424</td>\n      <td>1.229921</td>\n      <td>3.885032</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>1730.000000</td>\n      <td>1729.000000</td>\n      <td>1729.000000</td>\n      <td>1.000000</td>\n      <td>4.890430e+05</td>\n      <td>4087.000000</td>\n      <td>4087.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>1.0</td>\n      <td>...</td>\n      <td>2790.000000</td>\n      <td>2802.000000</td>\n      <td>2805.000000</td>\n      <td>2802.000000</td>\n      <td>2802.000000</td>\n      <td>2802.000000</td>\n      <td>2983.000000</td>\n      <td>1.040000</td>\n      <td>1.730000</td>\n      <td>1.170000</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>6015.000000</td>\n      <td>4769.000000</td>\n      <td>4769.000000</td>\n      <td>11.000000</td>\n      <td>8.572140e+05</td>\n      <td>8530.000000</td>\n      <td>8530.000000</td>\n      <td>1.000000</td>\n      <td>0.000000</td>\n      <td>1.0</td>\n      <td>...</td>\n      <td>32569.000000</td>\n      <td>30849.000000</td>\n      <td>30628.000000</td>\n      <td>30983.000000</td>\n      <td>32566.000000</td>\n      <td>30956.000000</td>\n      <td>30960.000000</td>\n      <td>1.670000</td>\n      <td>3.300000</td>\n      <td>2.600000</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>9424.000000</td>\n      <td>7809.000000</td>\n      <td>7809.000000</td>\n      <td>19.000000</td>\n      <td>1.239632e+06</td>\n      <td>8667.000000</td>\n      <td>8674.000000</td>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>1.0</td>\n      <td>...</td>\n      <td>41559.000000</td>\n      <td>39854.000000</td>\n      <td>39834.000000</td>\n      <td>41816.000000</td>\n      <td>41364.000000</td>\n      <td>40758.000000</td>\n      <td>40203.000000</td>\n      <td>2.100000</td>\n      <td>3.500000</td>\n      <td>3.600000</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>13201.000000</td>\n      <td>10257.000000</td>\n      <td>10257.000000</td>\n      <td>29.000000</td>\n      <td>1.724026e+06</td>\n      <td>9869.000000</td>\n      <td>9873.000000</td>\n      <td>2.000000</td>\n      <td>2.000000</td>\n      <td>1.0</td>\n      <td>...</td>\n      <td>154938.000000</td>\n      <td>141699.000000</td>\n      <td>127945.000000</td>\n      <td>155129.000000</td>\n      <td>154249.000000</td>\n      <td>154856.000000</td>\n      <td>147951.000000</td>\n      <td>2.750000</td>\n      <td>3.800000</td>\n      <td>5.250000</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>24557.000000</td>\n      <td>21518.000000</td>\n      <td>21518.000000</td>\n      <td>38.000000</td>\n      <td>2.060644e+06</td>\n      <td>208931.000000</td>\n      <td>208931.000000</td>\n      <td>10.000000</td>\n      <td>8.000000</td>\n      <td>1.0</td>\n      <td>...</td>\n      <td>681383.000000</td>\n      <td>680905.000000</td>\n      <td>701154.000000</td>\n      <td>704523.000000</td>\n      <td>684978.000000</td>\n      <td>722766.000000</td>\n      <td>725718.000000</td>\n      <td>15.000000</td>\n      <td>17.000000</td>\n      <td>41.000000</td>\n    </tr>\n  </tbody>\n</table>\n<p>8 rows \u00d7 78 columns</p>\n</div>",
   "text/plain": "                 id    country_id     league_id        stage  match_api_id  \\\ncount   2505.000000   2505.000000   2505.000000  2505.000000  2.505000e+03   \nmean   11050.111377   9471.790419   9471.790419    19.441118  1.273188e+06   \nstd     6971.115807   6923.153417   6923.153417    10.744260  4.837662e+05   \nmin     1730.000000   1729.000000   1729.000000     1.000000  4.890430e+05   \n25%     6015.000000   4769.000000   4769.000000    11.000000  8.572140e+05   \n50%     9424.000000   7809.000000   7809.000000    19.000000  1.239632e+06   \n75%    13201.000000  10257.000000  10257.000000    29.000000  1.724026e+06   \nmax    24557.000000  21518.000000  21518.000000    38.000000  2.060644e+06   \n\n       home_team_api_id  away_team_api_id  home_team_goal  away_team_goal  \\\ncount       2505.000000       2505.000000     2505.000000     2505.000000   \nmean        9326.698204       9268.572854        1.543313        1.144112   \nstd         6368.675817       4958.688585        1.288241        1.113556   \nmin         4087.000000       4087.000000        0.000000        0.000000   \n25%         8530.000000       8530.000000        1.000000        0.000000   \n50%         8667.000000       8674.000000        1.000000        1.000000   \n75%         9869.000000       9873.000000        2.000000        2.000000   \nmax       208931.000000     208931.000000       10.000000        8.000000   \n\n       home_player_X1  ...  away_player_5  away_player_6  away_player_7  \\\ncount          2505.0  ...    2505.000000    2505.000000    2505.000000   \nmean              1.0  ...  103986.502994   95100.075050   91516.336527   \nstd               0.0  ...  112696.240000  106123.109905  103938.792275   \nmin               1.0  ...    2790.000000    2802.000000    2805.000000   \n25%               1.0  ...   32569.000000   30849.000000   30628.000000   \n50%               1.0  ...   41559.000000   39854.000000   39834.000000   \n75%               1.0  ...  154938.000000  141699.000000  127945.000000   \nmax               1.0  ...  681383.000000  680905.000000  701154.000000   \n\n       away_player_8  away_player_9  away_player_10  away_player_11  \\\ncount    2505.000000    2505.000000     2505.000000     2505.000000   \nmean   105081.580439  105003.107784   102028.245509    96260.831537   \nstd    114656.410740  115588.861261   113190.241096   108851.544990   \nmin      2802.000000    2802.000000     2802.000000     2983.000000   \n25%     30983.000000   32566.000000    30956.000000    30960.000000   \n50%     41816.000000   41364.000000    40758.000000    40203.000000   \n75%    155129.000000  154249.000000   154856.000000   147951.000000   \nmax    704523.000000  684978.000000   722766.000000   725718.000000   \n\n             B365H        B365D        B365A  \ncount  2505.000000  2505.000000  2505.000000  \nmean      2.558671     3.857697     4.738930  \nstd       1.618424     1.229921     3.885032  \nmin       1.040000     1.730000     1.170000  \n25%       1.670000     3.300000     2.600000  \n50%       2.100000     3.500000     3.600000  \n75%       2.750000     3.800000     5.250000  \nmax      15.000000    17.000000    41.000000  \n\n[8 rows x 78 columns]"
  },
  "execution_count": 146,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### Team Stats - Team Attributes

```{.python .input  n=147}
teams_stats.describe()
```

```{.json .output n=147}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>team_fifa_api_id</th>\n      <th>team_api_id</th>\n      <th>buildUpPlaySpeed</th>\n      <th>buildUpPlayDribbling</th>\n      <th>buildUpPlayPassing</th>\n      <th>chanceCreationPassing</th>\n      <th>chanceCreationCrossing</th>\n      <th>chanceCreationShooting</th>\n      <th>defencePressure</th>\n      <th>defenceAggression</th>\n      <th>defenceTeamWidth</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>489.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>729.500000</td>\n      <td>17706.982167</td>\n      <td>9995.727023</td>\n      <td>52.462277</td>\n      <td>48.607362</td>\n      <td>48.490398</td>\n      <td>52.165295</td>\n      <td>53.731824</td>\n      <td>53.969136</td>\n      <td>46.017147</td>\n      <td>49.251029</td>\n      <td>52.185871</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>421.032659</td>\n      <td>39179.857739</td>\n      <td>13264.869900</td>\n      <td>11.545869</td>\n      <td>9.678290</td>\n      <td>10.896101</td>\n      <td>10.360793</td>\n      <td>11.086796</td>\n      <td>10.327566</td>\n      <td>10.227225</td>\n      <td>9.738028</td>\n      <td>9.574712</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>1601.000000</td>\n      <td>20.000000</td>\n      <td>24.000000</td>\n      <td>20.000000</td>\n      <td>21.000000</td>\n      <td>20.000000</td>\n      <td>22.000000</td>\n      <td>23.000000</td>\n      <td>24.000000</td>\n      <td>29.000000</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>365.250000</td>\n      <td>110.000000</td>\n      <td>8457.750000</td>\n      <td>45.000000</td>\n      <td>42.000000</td>\n      <td>40.000000</td>\n      <td>46.000000</td>\n      <td>47.000000</td>\n      <td>48.000000</td>\n      <td>39.000000</td>\n      <td>44.000000</td>\n      <td>47.000000</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>729.500000</td>\n      <td>485.000000</td>\n      <td>8674.000000</td>\n      <td>52.000000</td>\n      <td>49.000000</td>\n      <td>50.000000</td>\n      <td>52.000000</td>\n      <td>53.000000</td>\n      <td>53.000000</td>\n      <td>45.000000</td>\n      <td>48.000000</td>\n      <td>52.000000</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>1093.750000</td>\n      <td>1900.000000</td>\n      <td>9904.000000</td>\n      <td>62.000000</td>\n      <td>55.000000</td>\n      <td>55.000000</td>\n      <td>59.000000</td>\n      <td>62.000000</td>\n      <td>61.000000</td>\n      <td>51.000000</td>\n      <td>55.000000</td>\n      <td>58.000000</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>1458.000000</td>\n      <td>112513.000000</td>\n      <td>274581.000000</td>\n      <td>80.000000</td>\n      <td>77.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>72.000000</td>\n      <td>72.000000</td>\n      <td>73.000000</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                id  team_fifa_api_id    team_api_id  buildUpPlaySpeed  \\\ncount  1458.000000       1458.000000    1458.000000       1458.000000   \nmean    729.500000      17706.982167    9995.727023         52.462277   \nstd     421.032659      39179.857739   13264.869900         11.545869   \nmin       1.000000          1.000000    1601.000000         20.000000   \n25%     365.250000        110.000000    8457.750000         45.000000   \n50%     729.500000        485.000000    8674.000000         52.000000   \n75%    1093.750000       1900.000000    9904.000000         62.000000   \nmax    1458.000000     112513.000000  274581.000000         80.000000   \n\n       buildUpPlayDribbling  buildUpPlayPassing  chanceCreationPassing  \\\ncount            489.000000         1458.000000            1458.000000   \nmean              48.607362           48.490398              52.165295   \nstd                9.678290           10.896101              10.360793   \nmin               24.000000           20.000000              21.000000   \n25%               42.000000           40.000000              46.000000   \n50%               49.000000           50.000000              52.000000   \n75%               55.000000           55.000000              59.000000   \nmax               77.000000           80.000000              80.000000   \n\n       chanceCreationCrossing  chanceCreationShooting  defencePressure  \\\ncount             1458.000000             1458.000000      1458.000000   \nmean                53.731824               53.969136        46.017147   \nstd                 11.086796               10.327566        10.227225   \nmin                 20.000000               22.000000        23.000000   \n25%                 47.000000               48.000000        39.000000   \n50%                 53.000000               53.000000        45.000000   \n75%                 62.000000               61.000000        51.000000   \nmax                 80.000000               80.000000        72.000000   \n\n       defenceAggression  defenceTeamWidth  \ncount        1458.000000       1458.000000  \nmean           49.251029         52.185871  \nstd             9.738028          9.574712  \nmin            24.000000         29.000000  \n25%            44.000000         47.000000  \n50%            48.000000         52.000000  \n75%            55.000000         58.000000  \nmax            72.000000         73.000000  "
  },
  "execution_count": 147,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Looking at the description of team attributes we can see that there are a lot of
values missing from the column buildUpPlayDribbling, and all the other values
seem to have the right amout of rows. This means that there are a lot of values
with 'Nan' on this column.

It's not ideal that we just drop those rows. Seems like the missing data on the
column is systematic - all of the missing values are in the same column - this
error could potentially bias our analysis.

One way to deal with missing values is **mean imputation**. If we know that the
values for a measurement fall in a certain range, we can fill in empty values
with the average of that measure.

```{.python .input  n=148}
teams_stats['buildUpPlayDribbling'].hist();
```

```{.json .output n=148}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

We can see that most buildUpPlayDribbling values fall within the 45 - 55 range,
so let's fill in these entries with the average measured buildUpPlaySpeed

```{.python .input  n=149}
build_up_play_drib_avg = teams_stats['buildUpPlayDribbling'].mean()
# mean imputation
teams_stats.loc[(teams_stats['buildUpPlayDribbling'].isnull()), 'buildUpPlayDribbling'] = build_up_play_drib_avg
# showing new values
teams_stats.loc[teams_stats['buildUpPlayDribbling'] == build_up_play_drib_avg].head()
```

```{.json .output n=149}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>team_fifa_api_id</th>\n      <th>team_api_id</th>\n      <th>date</th>\n      <th>buildUpPlaySpeed</th>\n      <th>buildUpPlaySpeedClass</th>\n      <th>buildUpPlayDribbling</th>\n      <th>buildUpPlayDribblingClass</th>\n      <th>buildUpPlayPassing</th>\n      <th>buildUpPlayPassingClass</th>\n      <th>...</th>\n      <th>chanceCreationShooting</th>\n      <th>chanceCreationShootingClass</th>\n      <th>chanceCreationPositioningClass</th>\n      <th>defencePressure</th>\n      <th>defencePressureClass</th>\n      <th>defenceAggression</th>\n      <th>defenceAggressionClass</th>\n      <th>defenceTeamWidth</th>\n      <th>defenceTeamWidthClass</th>\n      <th>defenceDefenderLineClass</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1</td>\n      <td>434</td>\n      <td>9930</td>\n      <td>2010-02-22 00:00:00</td>\n      <td>60</td>\n      <td>Balanced</td>\n      <td>48.607362</td>\n      <td>Little</td>\n      <td>50</td>\n      <td>Mixed</td>\n      <td>...</td>\n      <td>55</td>\n      <td>Normal</td>\n      <td>Organised</td>\n      <td>50</td>\n      <td>Medium</td>\n      <td>55</td>\n      <td>Press</td>\n      <td>45</td>\n      <td>Normal</td>\n      <td>Cover</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>4</td>\n      <td>77</td>\n      <td>8485</td>\n      <td>2010-02-22 00:00:00</td>\n      <td>70</td>\n      <td>Fast</td>\n      <td>48.607362</td>\n      <td>Little</td>\n      <td>70</td>\n      <td>Long</td>\n      <td>...</td>\n      <td>70</td>\n      <td>Lots</td>\n      <td>Organised</td>\n      <td>60</td>\n      <td>Medium</td>\n      <td>70</td>\n      <td>Double</td>\n      <td>70</td>\n      <td>Wide</td>\n      <td>Cover</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>5</td>\n      <td>77</td>\n      <td>8485</td>\n      <td>2011-02-22 00:00:00</td>\n      <td>47</td>\n      <td>Balanced</td>\n      <td>48.607362</td>\n      <td>Little</td>\n      <td>52</td>\n      <td>Mixed</td>\n      <td>...</td>\n      <td>52</td>\n      <td>Normal</td>\n      <td>Organised</td>\n      <td>47</td>\n      <td>Medium</td>\n      <td>47</td>\n      <td>Press</td>\n      <td>52</td>\n      <td>Normal</td>\n      <td>Cover</td>\n    </tr>\n    <tr>\n      <th>5</th>\n      <td>6</td>\n      <td>77</td>\n      <td>8485</td>\n      <td>2012-02-22 00:00:00</td>\n      <td>58</td>\n      <td>Balanced</td>\n      <td>48.607362</td>\n      <td>Little</td>\n      <td>62</td>\n      <td>Mixed</td>\n      <td>...</td>\n      <td>55</td>\n      <td>Normal</td>\n      <td>Organised</td>\n      <td>40</td>\n      <td>Medium</td>\n      <td>40</td>\n      <td>Press</td>\n      <td>60</td>\n      <td>Normal</td>\n      <td>Cover</td>\n    </tr>\n    <tr>\n      <th>6</th>\n      <td>7</td>\n      <td>77</td>\n      <td>8485</td>\n      <td>2013-09-20 00:00:00</td>\n      <td>62</td>\n      <td>Balanced</td>\n      <td>48.607362</td>\n      <td>Little</td>\n      <td>45</td>\n      <td>Mixed</td>\n      <td>...</td>\n      <td>55</td>\n      <td>Normal</td>\n      <td>Organised</td>\n      <td>42</td>\n      <td>Medium</td>\n      <td>42</td>\n      <td>Press</td>\n      <td>60</td>\n      <td>Normal</td>\n      <td>Cover</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 25 columns</p>\n</div>",
   "text/plain": "   id  team_fifa_api_id  team_api_id                 date  buildUpPlaySpeed  \\\n0   1               434         9930  2010-02-22 00:00:00                60   \n3   4                77         8485  2010-02-22 00:00:00                70   \n4   5                77         8485  2011-02-22 00:00:00                47   \n5   6                77         8485  2012-02-22 00:00:00                58   \n6   7                77         8485  2013-09-20 00:00:00                62   \n\n  buildUpPlaySpeedClass  buildUpPlayDribbling buildUpPlayDribblingClass  \\\n0              Balanced             48.607362                    Little   \n3                  Fast             48.607362                    Little   \n4              Balanced             48.607362                    Little   \n5              Balanced             48.607362                    Little   \n6              Balanced             48.607362                    Little   \n\n   buildUpPlayPassing buildUpPlayPassingClass  ... chanceCreationShooting  \\\n0                  50                   Mixed  ...                     55   \n3                  70                    Long  ...                     70   \n4                  52                   Mixed  ...                     52   \n5                  62                   Mixed  ...                     55   \n6                  45                   Mixed  ...                     55   \n\n   chanceCreationShootingClass chanceCreationPositioningClass  \\\n0                       Normal                      Organised   \n3                         Lots                      Organised   \n4                       Normal                      Organised   \n5                       Normal                      Organised   \n6                       Normal                      Organised   \n\n   defencePressure defencePressureClass  defenceAggression  \\\n0               50               Medium                 55   \n3               60               Medium                 70   \n4               47               Medium                 47   \n5               40               Medium                 40   \n6               42               Medium                 42   \n\n  defenceAggressionClass defenceTeamWidth  defenceTeamWidthClass  \\\n0                  Press               45                 Normal   \n3                 Double               70                   Wide   \n4                  Press               52                 Normal   \n5                  Press               60                 Normal   \n6                  Press               60                 Normal   \n\n  defenceDefenderLineClass  \n0                    Cover  \n3                    Cover  \n4                    Cover  \n5                    Cover  \n6                    Cover  \n\n[5 rows x 25 columns]"
  },
  "execution_count": 149,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=150}
teams_stats.loc[(teams_stats['buildUpPlayDribbling'].isnull())]
```

```{.json .output n=150}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>team_fifa_api_id</th>\n      <th>team_api_id</th>\n      <th>date</th>\n      <th>buildUpPlaySpeed</th>\n      <th>buildUpPlaySpeedClass</th>\n      <th>buildUpPlayDribbling</th>\n      <th>buildUpPlayDribblingClass</th>\n      <th>buildUpPlayPassing</th>\n      <th>buildUpPlayPassingClass</th>\n      <th>...</th>\n      <th>chanceCreationShooting</th>\n      <th>chanceCreationShootingClass</th>\n      <th>chanceCreationPositioningClass</th>\n      <th>defencePressure</th>\n      <th>defencePressureClass</th>\n      <th>defenceAggression</th>\n      <th>defenceAggressionClass</th>\n      <th>defenceTeamWidth</th>\n      <th>defenceTeamWidthClass</th>\n      <th>defenceDefenderLineClass</th>\n    </tr>\n  </thead>\n  <tbody>\n  </tbody>\n</table>\n<p>0 rows \u00d7 25 columns</p>\n</div>",
   "text/plain": "Empty DataFrame\nColumns: [id, team_fifa_api_id, team_api_id, date, buildUpPlaySpeed, buildUpPlaySpeedClass, buildUpPlayDribbling, buildUpPlayDribblingClass, buildUpPlayPassing, buildUpPlayPassingClass, buildUpPlayPositioningClass, chanceCreationPassing, chanceCreationPassingClass, chanceCreationCrossing, chanceCreationCrossingClass, chanceCreationShooting, chanceCreationShootingClass, chanceCreationPositioningClass, defencePressure, defencePressureClass, defenceAggression, defenceAggressionClass, defenceTeamWidth, defenceTeamWidthClass, defenceDefenderLineClass]\nIndex: []\n\n[0 rows x 25 columns]"
  },
  "execution_count": 150,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Having done the **mean imputation** for team_attributes we can see that there
are no longer missing values for the buildUpPlayDribbling. After that, we
decided to select only continuous data, i.e, select only columns that "store"
numerical values that we will provide to the input of the supervised learning
models.

```{.python .input  n=151}
teams_stats.drop(['buildUpPlaySpeedClass', 'buildUpPlayDribblingClass', 'buildUpPlayPassingClass', 
    'buildUpPlayPositioningClass', 'chanceCreationPassingClass', 'chanceCreationCrossingClass',  
    'chanceCreationShootingClass','chanceCreationPositioningClass','defencePressureClass', 'defenceAggressionClass', 
    'defenceTeamWidthClass','defenceDefenderLineClass'], inplace = True, axis = 1)

teams_stats.describe()
```

```{.json .output n=151}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>team_fifa_api_id</th>\n      <th>team_api_id</th>\n      <th>buildUpPlaySpeed</th>\n      <th>buildUpPlayDribbling</th>\n      <th>buildUpPlayPassing</th>\n      <th>chanceCreationPassing</th>\n      <th>chanceCreationCrossing</th>\n      <th>chanceCreationShooting</th>\n      <th>defencePressure</th>\n      <th>defenceAggression</th>\n      <th>defenceTeamWidth</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>729.500000</td>\n      <td>17706.982167</td>\n      <td>9995.727023</td>\n      <td>52.462277</td>\n      <td>48.607362</td>\n      <td>48.490398</td>\n      <td>52.165295</td>\n      <td>53.731824</td>\n      <td>53.969136</td>\n      <td>46.017147</td>\n      <td>49.251029</td>\n      <td>52.185871</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>421.032659</td>\n      <td>39179.857739</td>\n      <td>13264.869900</td>\n      <td>11.545869</td>\n      <td>5.601170</td>\n      <td>10.896101</td>\n      <td>10.360793</td>\n      <td>11.086796</td>\n      <td>10.327566</td>\n      <td>10.227225</td>\n      <td>9.738028</td>\n      <td>9.574712</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>1601.000000</td>\n      <td>20.000000</td>\n      <td>24.000000</td>\n      <td>20.000000</td>\n      <td>21.000000</td>\n      <td>20.000000</td>\n      <td>22.000000</td>\n      <td>23.000000</td>\n      <td>24.000000</td>\n      <td>29.000000</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>365.250000</td>\n      <td>110.000000</td>\n      <td>8457.750000</td>\n      <td>45.000000</td>\n      <td>48.607362</td>\n      <td>40.000000</td>\n      <td>46.000000</td>\n      <td>47.000000</td>\n      <td>48.000000</td>\n      <td>39.000000</td>\n      <td>44.000000</td>\n      <td>47.000000</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>729.500000</td>\n      <td>485.000000</td>\n      <td>8674.000000</td>\n      <td>52.000000</td>\n      <td>48.607362</td>\n      <td>50.000000</td>\n      <td>52.000000</td>\n      <td>53.000000</td>\n      <td>53.000000</td>\n      <td>45.000000</td>\n      <td>48.000000</td>\n      <td>52.000000</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>1093.750000</td>\n      <td>1900.000000</td>\n      <td>9904.000000</td>\n      <td>62.000000</td>\n      <td>48.607362</td>\n      <td>55.000000</td>\n      <td>59.000000</td>\n      <td>62.000000</td>\n      <td>61.000000</td>\n      <td>51.000000</td>\n      <td>55.000000</td>\n      <td>58.000000</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>1458.000000</td>\n      <td>112513.000000</td>\n      <td>274581.000000</td>\n      <td>80.000000</td>\n      <td>77.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>72.000000</td>\n      <td>72.000000</td>\n      <td>73.000000</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                id  team_fifa_api_id    team_api_id  buildUpPlaySpeed  \\\ncount  1458.000000       1458.000000    1458.000000       1458.000000   \nmean    729.500000      17706.982167    9995.727023         52.462277   \nstd     421.032659      39179.857739   13264.869900         11.545869   \nmin       1.000000          1.000000    1601.000000         20.000000   \n25%     365.250000        110.000000    8457.750000         45.000000   \n50%     729.500000        485.000000    8674.000000         52.000000   \n75%    1093.750000       1900.000000    9904.000000         62.000000   \nmax    1458.000000     112513.000000  274581.000000         80.000000   \n\n       buildUpPlayDribbling  buildUpPlayPassing  chanceCreationPassing  \\\ncount           1458.000000         1458.000000            1458.000000   \nmean              48.607362           48.490398              52.165295   \nstd                5.601170           10.896101              10.360793   \nmin               24.000000           20.000000              21.000000   \n25%               48.607362           40.000000              46.000000   \n50%               48.607362           50.000000              52.000000   \n75%               48.607362           55.000000              59.000000   \nmax               77.000000           80.000000              80.000000   \n\n       chanceCreationCrossing  chanceCreationShooting  defencePressure  \\\ncount             1458.000000             1458.000000      1458.000000   \nmean                53.731824               53.969136        46.017147   \nstd                 11.086796               10.327566        10.227225   \nmin                 20.000000               22.000000        23.000000   \n25%                 47.000000               48.000000        39.000000   \n50%                 53.000000               53.000000        45.000000   \n75%                 62.000000               61.000000        51.000000   \nmax                 80.000000               80.000000        72.000000   \n\n       defenceAggression  defenceTeamWidth  \ncount        1458.000000       1458.000000  \nmean           49.251029         52.185871  \nstd             9.738028          9.574712  \nmin            24.000000         29.000000  \n25%            44.000000         47.000000  \n50%            48.000000         52.000000  \n75%            55.000000         58.000000  \nmax            72.000000         73.000000  "
  },
  "execution_count": 151,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### Team Stats - Shots
After cleaning the team attributes data we need to consider adding some more
stats to each match. We will start by adding the average of the number of shots
per team. The number of shots consists on the sum of the shots on target and the
shots of target. After merging all the information to teams_stats we have to
analyse the data again.

```{.python .input  n=152}
shots_off = pd.read_csv("../dataset/shotoff_detail.csv")
shots_on = pd.read_csv("../dataset/shoton_detail.csv")
shots = pd.concat([shots_off[['match_id', 'team']], shots_on[['match_id', 'team']]])

total_shots = shots["team"].value_counts()
total_matches = shots.drop_duplicates(['match_id', 'team'])["team"].value_counts()

for index, n_shots in total_shots.iteritems():
    n_matches = total_matches[index]
    avg_shots = n_shots / n_matches
    teams_stats.loc[teams_stats['team_api_id'] == index, 'avg_shots'] = avg_shots
    
teams_stats.describe()
```

```{.json .output n=152}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "C:\\Python\\Python38\\lib\\site-packages\\IPython\\core\\interactiveshell.py:3062: DtypeWarning: Columns (0,14) have mixed types.Specify dtype option on import or set low_memory=False.\n  has_raised = await self.run_ast_nodes(code_ast.body, cell_name,\nC:\\Python\\Python38\\lib\\site-packages\\IPython\\core\\interactiveshell.py:3062: DtypeWarning: Columns (1,7,16) have mixed types.Specify dtype option on import or set low_memory=False.\n  has_raised = await self.run_ast_nodes(code_ast.body, cell_name,\n"
 },
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>team_fifa_api_id</th>\n      <th>team_api_id</th>\n      <th>buildUpPlaySpeed</th>\n      <th>buildUpPlayDribbling</th>\n      <th>buildUpPlayPassing</th>\n      <th>chanceCreationPassing</th>\n      <th>chanceCreationCrossing</th>\n      <th>chanceCreationShooting</th>\n      <th>defencePressure</th>\n      <th>defenceAggression</th>\n      <th>defenceTeamWidth</th>\n      <th>avg_shots</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>997.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>729.500000</td>\n      <td>17706.982167</td>\n      <td>9995.727023</td>\n      <td>52.462277</td>\n      <td>48.607362</td>\n      <td>48.490398</td>\n      <td>52.165295</td>\n      <td>53.731824</td>\n      <td>53.969136</td>\n      <td>46.017147</td>\n      <td>49.251029</td>\n      <td>52.185871</td>\n      <td>10.579949</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>421.032659</td>\n      <td>39179.857739</td>\n      <td>13264.869900</td>\n      <td>11.545869</td>\n      <td>5.601170</td>\n      <td>10.896101</td>\n      <td>10.360793</td>\n      <td>11.086796</td>\n      <td>10.327566</td>\n      <td>10.227225</td>\n      <td>9.738028</td>\n      <td>9.574712</td>\n      <td>1.808530</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>1601.000000</td>\n      <td>20.000000</td>\n      <td>24.000000</td>\n      <td>20.000000</td>\n      <td>21.000000</td>\n      <td>20.000000</td>\n      <td>22.000000</td>\n      <td>23.000000</td>\n      <td>24.000000</td>\n      <td>29.000000</td>\n      <td>3.000000</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>365.250000</td>\n      <td>110.000000</td>\n      <td>8457.750000</td>\n      <td>45.000000</td>\n      <td>48.607362</td>\n      <td>40.000000</td>\n      <td>46.000000</td>\n      <td>47.000000</td>\n      <td>48.000000</td>\n      <td>39.000000</td>\n      <td>44.000000</td>\n      <td>47.000000</td>\n      <td>9.470588</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>729.500000</td>\n      <td>485.000000</td>\n      <td>8674.000000</td>\n      <td>52.000000</td>\n      <td>48.607362</td>\n      <td>50.000000</td>\n      <td>52.000000</td>\n      <td>53.000000</td>\n      <td>53.000000</td>\n      <td>45.000000</td>\n      <td>48.000000</td>\n      <td>52.000000</td>\n      <td>10.428571</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>1093.750000</td>\n      <td>1900.000000</td>\n      <td>9904.000000</td>\n      <td>62.000000</td>\n      <td>48.607362</td>\n      <td>55.000000</td>\n      <td>59.000000</td>\n      <td>62.000000</td>\n      <td>61.000000</td>\n      <td>51.000000</td>\n      <td>55.000000</td>\n      <td>58.000000</td>\n      <td>11.376812</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>1458.000000</td>\n      <td>112513.000000</td>\n      <td>274581.000000</td>\n      <td>80.000000</td>\n      <td>77.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>72.000000</td>\n      <td>72.000000</td>\n      <td>73.000000</td>\n      <td>16.000000</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                id  team_fifa_api_id    team_api_id  buildUpPlaySpeed  \\\ncount  1458.000000       1458.000000    1458.000000       1458.000000   \nmean    729.500000      17706.982167    9995.727023         52.462277   \nstd     421.032659      39179.857739   13264.869900         11.545869   \nmin       1.000000          1.000000    1601.000000         20.000000   \n25%     365.250000        110.000000    8457.750000         45.000000   \n50%     729.500000        485.000000    8674.000000         52.000000   \n75%    1093.750000       1900.000000    9904.000000         62.000000   \nmax    1458.000000     112513.000000  274581.000000         80.000000   \n\n       buildUpPlayDribbling  buildUpPlayPassing  chanceCreationPassing  \\\ncount           1458.000000         1458.000000            1458.000000   \nmean              48.607362           48.490398              52.165295   \nstd                5.601170           10.896101              10.360793   \nmin               24.000000           20.000000              21.000000   \n25%               48.607362           40.000000              46.000000   \n50%               48.607362           50.000000              52.000000   \n75%               48.607362           55.000000              59.000000   \nmax               77.000000           80.000000              80.000000   \n\n       chanceCreationCrossing  chanceCreationShooting  defencePressure  \\\ncount             1458.000000             1458.000000      1458.000000   \nmean                53.731824               53.969136        46.017147   \nstd                 11.086796               10.327566        10.227225   \nmin                 20.000000               22.000000        23.000000   \n25%                 47.000000               48.000000        39.000000   \n50%                 53.000000               53.000000        45.000000   \n75%                 62.000000               61.000000        51.000000   \nmax                 80.000000               80.000000        72.000000   \n\n       defenceAggression  defenceTeamWidth   avg_shots  \ncount        1458.000000       1458.000000  997.000000  \nmean           49.251029         52.185871   10.579949  \nstd             9.738028          9.574712    1.808530  \nmin            24.000000         29.000000    3.000000  \n25%            44.000000         47.000000    9.470588  \n50%            48.000000         52.000000   10.428571  \n75%            55.000000         58.000000   11.376812  \nmax            72.000000         73.000000   16.000000  "
  },
  "execution_count": 152,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

As we can see, there are a lot of Nan values on the avg_shots column. This
represents teams that did not have shots data on this dataset. Instead of
removing thoose rows, and give less input to our models we need again to do
**mean imputation** and deal with these values.

```{.python .input  n=153}
teams_stats['avg_shots'].hist();
```

```{.json .output n=153}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

We can see that most avg_shots values fall within the 7 - 14 range, so let's
fill in these entries with the average measured avg_shots

```{.python .input  n=154}
shots_avg_team_avg = teams_stats['avg_shots'].mean()
# mean imputation
teams_stats.loc[(teams_stats['avg_shots'].isnull()), 'avg_shots'] = shots_avg_team_avg
# showing new values
teams_stats.describe()
```

```{.json .output n=154}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>team_fifa_api_id</th>\n      <th>team_api_id</th>\n      <th>buildUpPlaySpeed</th>\n      <th>buildUpPlayDribbling</th>\n      <th>buildUpPlayPassing</th>\n      <th>chanceCreationPassing</th>\n      <th>chanceCreationCrossing</th>\n      <th>chanceCreationShooting</th>\n      <th>defencePressure</th>\n      <th>defenceAggression</th>\n      <th>defenceTeamWidth</th>\n      <th>avg_shots</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>729.500000</td>\n      <td>17706.982167</td>\n      <td>9995.727023</td>\n      <td>52.462277</td>\n      <td>48.607362</td>\n      <td>48.490398</td>\n      <td>52.165295</td>\n      <td>53.731824</td>\n      <td>53.969136</td>\n      <td>46.017147</td>\n      <td>49.251029</td>\n      <td>52.185871</td>\n      <td>10.579949</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>421.032659</td>\n      <td>39179.857739</td>\n      <td>13264.869900</td>\n      <td>11.545869</td>\n      <td>5.601170</td>\n      <td>10.896101</td>\n      <td>10.360793</td>\n      <td>11.086796</td>\n      <td>10.327566</td>\n      <td>10.227225</td>\n      <td>9.738028</td>\n      <td>9.574712</td>\n      <td>1.495291</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>1601.000000</td>\n      <td>20.000000</td>\n      <td>24.000000</td>\n      <td>20.000000</td>\n      <td>21.000000</td>\n      <td>20.000000</td>\n      <td>22.000000</td>\n      <td>23.000000</td>\n      <td>24.000000</td>\n      <td>29.000000</td>\n      <td>3.000000</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>365.250000</td>\n      <td>110.000000</td>\n      <td>8457.750000</td>\n      <td>45.000000</td>\n      <td>48.607362</td>\n      <td>40.000000</td>\n      <td>46.000000</td>\n      <td>47.000000</td>\n      <td>48.000000</td>\n      <td>39.000000</td>\n      <td>44.000000</td>\n      <td>47.000000</td>\n      <td>9.947368</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>729.500000</td>\n      <td>485.000000</td>\n      <td>8674.000000</td>\n      <td>52.000000</td>\n      <td>48.607362</td>\n      <td>50.000000</td>\n      <td>52.000000</td>\n      <td>53.000000</td>\n      <td>53.000000</td>\n      <td>45.000000</td>\n      <td>48.000000</td>\n      <td>52.000000</td>\n      <td>10.579949</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>1093.750000</td>\n      <td>1900.000000</td>\n      <td>9904.000000</td>\n      <td>62.000000</td>\n      <td>48.607362</td>\n      <td>55.000000</td>\n      <td>59.000000</td>\n      <td>62.000000</td>\n      <td>61.000000</td>\n      <td>51.000000</td>\n      <td>55.000000</td>\n      <td>58.000000</td>\n      <td>10.960000</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>1458.000000</td>\n      <td>112513.000000</td>\n      <td>274581.000000</td>\n      <td>80.000000</td>\n      <td>77.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>72.000000</td>\n      <td>72.000000</td>\n      <td>73.000000</td>\n      <td>16.000000</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                id  team_fifa_api_id    team_api_id  buildUpPlaySpeed  \\\ncount  1458.000000       1458.000000    1458.000000       1458.000000   \nmean    729.500000      17706.982167    9995.727023         52.462277   \nstd     421.032659      39179.857739   13264.869900         11.545869   \nmin       1.000000          1.000000    1601.000000         20.000000   \n25%     365.250000        110.000000    8457.750000         45.000000   \n50%     729.500000        485.000000    8674.000000         52.000000   \n75%    1093.750000       1900.000000    9904.000000         62.000000   \nmax    1458.000000     112513.000000  274581.000000         80.000000   \n\n       buildUpPlayDribbling  buildUpPlayPassing  chanceCreationPassing  \\\ncount           1458.000000         1458.000000            1458.000000   \nmean              48.607362           48.490398              52.165295   \nstd                5.601170           10.896101              10.360793   \nmin               24.000000           20.000000              21.000000   \n25%               48.607362           40.000000              46.000000   \n50%               48.607362           50.000000              52.000000   \n75%               48.607362           55.000000              59.000000   \nmax               77.000000           80.000000              80.000000   \n\n       chanceCreationCrossing  chanceCreationShooting  defencePressure  \\\ncount             1458.000000             1458.000000      1458.000000   \nmean                53.731824               53.969136        46.017147   \nstd                 11.086796               10.327566        10.227225   \nmin                 20.000000               22.000000        23.000000   \n25%                 47.000000               48.000000        39.000000   \n50%                 53.000000               53.000000        45.000000   \n75%                 62.000000               61.000000        51.000000   \nmax                 80.000000               80.000000        72.000000   \n\n       defenceAggression  defenceTeamWidth    avg_shots  \ncount        1458.000000       1458.000000  1458.000000  \nmean           49.251029         52.185871    10.579949  \nstd             9.738028          9.574712     1.495291  \nmin            24.000000         29.000000     3.000000  \n25%            44.000000         47.000000     9.947368  \n50%            48.000000         52.000000    10.579949  \n75%            55.000000         58.000000    10.960000  \nmax            72.000000         73.000000    16.000000  "
  },
  "execution_count": 154,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=155}
teams_stats.loc[(teams_stats['avg_shots'].isnull())]
```

```{.json .output n=155}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>team_fifa_api_id</th>\n      <th>team_api_id</th>\n      <th>date</th>\n      <th>buildUpPlaySpeed</th>\n      <th>buildUpPlayDribbling</th>\n      <th>buildUpPlayPassing</th>\n      <th>chanceCreationPassing</th>\n      <th>chanceCreationCrossing</th>\n      <th>chanceCreationShooting</th>\n      <th>defencePressure</th>\n      <th>defenceAggression</th>\n      <th>defenceTeamWidth</th>\n      <th>avg_shots</th>\n    </tr>\n  </thead>\n  <tbody>\n  </tbody>\n</table>\n</div>",
   "text/plain": "Empty DataFrame\nColumns: [id, team_fifa_api_id, team_api_id, date, buildUpPlaySpeed, buildUpPlayDribbling, buildUpPlayPassing, chanceCreationPassing, chanceCreationCrossing, chanceCreationShooting, defencePressure, defenceAggression, defenceTeamWidth, avg_shots]\nIndex: []"
  },
  "execution_count": 155,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Having done the **mean imputation** for team_attributes we can see that there
are no longer missing values for the avg_shots.

### Team Stats - Possession

We will now add another stat, the ball possession. One of the more important
statistics to predict the match results. If we look closely the most dominating
teams usually control the ball possession very easily. We can see first that the
csv have repeated values for the ball possession, for each match, based on the
elapsed time of the match. We need to remove all the values that do not refer to
the 90 minutes mark of the elapsed time.

```{.python .input  n=156}
# possessions read, cleanup and merge
possessions_data = pd.read_csv("../dataset/possession_detail.csv")
last_possessions = possessions_data.sort_values(['elapsed'], ascending=False).drop_duplicates(subset=['match_id'])
last_possessions = last_possessions[['match_id', 'homepos', 'awaypos']]
```

```{.json .output n=156}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "C:\\Python\\Python38\\lib\\site-packages\\IPython\\core\\interactiveshell.py:3062: DtypeWarning: Columns (1,2,7) have mixed types.Specify dtype option on import or set low_memory=False.\n  has_raised = await self.run_ast_nodes(code_ast.body, cell_name,\n"
 }
]
```

After reading it, we need to see if the number of possession data we have
available is enough to be considered

```{.python .input  n=157}
# get the ids of the home_team and away_team to be able to join with teams later
possessions = pd.DataFrame(columns=['team', 'possession', 'match'])
for index, row in last_possessions.iterrows():
    match = matches.loc[matches['id'] == row['match_id'], ['home_team_api_id', 'away_team_api_id']]
    if match.empty:
        continue
    hometeam = match['home_team_api_id'].values[0]
    awayteam = match['away_team_api_id'].values[0]
    possessions = possessions.append({'team': hometeam, 'possession': row['homepos'], 'match': row['match_id']}, ignore_index=True)
    possessions = possessions.append({'team': awayteam, 'possession': row['awaypos'], 'match': row['match_id']}, ignore_index=True)

total_possessions = possessions.groupby(by=['team'])['possession'].sum()
total_matches = possessions.drop_duplicates(['team', 'match'])["team"].value_counts()
    
total_possessions.to_frame().describe()
```

```{.json .output n=157}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>possession</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>178.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>4728.651685</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>4268.085611</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>38.000000</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>1650.750000</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>3690.500000</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>6623.500000</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>17221.000000</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "         possession\ncount    178.000000\nmean    4728.651685\nstd     4268.085611\nmin       38.000000\n25%     1650.750000\n50%     3690.500000\n75%     6623.500000\nmax    17221.000000"
  },
  "execution_count": 157,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Since the number of average possession regarding the number of viable matches is
very low it doesn't make any sense to do **mean imputation** in this instance.
After carefully consideration, we decided to scrap this attribute, even though
this attribute as a very important meaning. This reflects the poor usability of
this dataset.

### Team Stats - Corners
We will try to add yet another feature. This is time the corners, also an
important measurement of domination in a football match. After merging all
corners data that were given to us we can see that there are some values
missing.

```{.python .input  n=158}
corners_data = pd.read_csv("../dataset/corner_detail.csv")
corners = corners_data[['match_id', 'team']]

total_corners = corners["team"].value_counts()
total_matches = corners.drop_duplicates(['match_id', 'team'])["team"].value_counts()

for index, n_corners in total_shots.iteritems():
    n_matches = total_matches[index]
    avg_corners = n_corners / n_matches
    teams_stats.loc[teams_stats['team_api_id'] == index, 'avg_corners'] = avg_corners
    
teams_stats.describe()
```

```{.json .output n=158}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "C:\\Python\\Python38\\lib\\site-packages\\IPython\\core\\interactiveshell.py:3062: DtypeWarning: Columns (15) have mixed types.Specify dtype option on import or set low_memory=False.\n  has_raised = await self.run_ast_nodes(code_ast.body, cell_name,\n"
 },
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>team_fifa_api_id</th>\n      <th>team_api_id</th>\n      <th>buildUpPlaySpeed</th>\n      <th>buildUpPlayDribbling</th>\n      <th>buildUpPlayPassing</th>\n      <th>chanceCreationPassing</th>\n      <th>chanceCreationCrossing</th>\n      <th>chanceCreationShooting</th>\n      <th>defencePressure</th>\n      <th>defenceAggression</th>\n      <th>defenceTeamWidth</th>\n      <th>avg_shots</th>\n      <th>avg_corners</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>997.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>729.500000</td>\n      <td>17706.982167</td>\n      <td>9995.727023</td>\n      <td>52.462277</td>\n      <td>48.607362</td>\n      <td>48.490398</td>\n      <td>52.165295</td>\n      <td>53.731824</td>\n      <td>53.969136</td>\n      <td>46.017147</td>\n      <td>49.251029</td>\n      <td>52.185871</td>\n      <td>10.579949</td>\n      <td>10.807166</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>421.032659</td>\n      <td>39179.857739</td>\n      <td>13264.869900</td>\n      <td>11.545869</td>\n      <td>5.601170</td>\n      <td>10.896101</td>\n      <td>10.360793</td>\n      <td>11.086796</td>\n      <td>10.327566</td>\n      <td>10.227225</td>\n      <td>9.738028</td>\n      <td>9.574712</td>\n      <td>1.495291</td>\n      <td>1.916837</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>1601.000000</td>\n      <td>20.000000</td>\n      <td>24.000000</td>\n      <td>20.000000</td>\n      <td>21.000000</td>\n      <td>20.000000</td>\n      <td>22.000000</td>\n      <td>23.000000</td>\n      <td>24.000000</td>\n      <td>29.000000</td>\n      <td>3.000000</td>\n      <td>3.000000</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>365.250000</td>\n      <td>110.000000</td>\n      <td>8457.750000</td>\n      <td>45.000000</td>\n      <td>48.607362</td>\n      <td>40.000000</td>\n      <td>46.000000</td>\n      <td>47.000000</td>\n      <td>48.000000</td>\n      <td>39.000000</td>\n      <td>44.000000</td>\n      <td>47.000000</td>\n      <td>9.947368</td>\n      <td>9.700000</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>729.500000</td>\n      <td>485.000000</td>\n      <td>8674.000000</td>\n      <td>52.000000</td>\n      <td>48.607362</td>\n      <td>50.000000</td>\n      <td>52.000000</td>\n      <td>53.000000</td>\n      <td>53.000000</td>\n      <td>45.000000</td>\n      <td>48.000000</td>\n      <td>52.000000</td>\n      <td>10.579949</td>\n      <td>10.591837</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>1093.750000</td>\n      <td>1900.000000</td>\n      <td>9904.000000</td>\n      <td>62.000000</td>\n      <td>48.607362</td>\n      <td>55.000000</td>\n      <td>59.000000</td>\n      <td>62.000000</td>\n      <td>61.000000</td>\n      <td>51.000000</td>\n      <td>55.000000</td>\n      <td>58.000000</td>\n      <td>10.960000</td>\n      <td>11.526316</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>1458.000000</td>\n      <td>112513.000000</td>\n      <td>274581.000000</td>\n      <td>80.000000</td>\n      <td>77.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>72.000000</td>\n      <td>72.000000</td>\n      <td>73.000000</td>\n      <td>16.000000</td>\n      <td>21.000000</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                id  team_fifa_api_id    team_api_id  buildUpPlaySpeed  \\\ncount  1458.000000       1458.000000    1458.000000       1458.000000   \nmean    729.500000      17706.982167    9995.727023         52.462277   \nstd     421.032659      39179.857739   13264.869900         11.545869   \nmin       1.000000          1.000000    1601.000000         20.000000   \n25%     365.250000        110.000000    8457.750000         45.000000   \n50%     729.500000        485.000000    8674.000000         52.000000   \n75%    1093.750000       1900.000000    9904.000000         62.000000   \nmax    1458.000000     112513.000000  274581.000000         80.000000   \n\n       buildUpPlayDribbling  buildUpPlayPassing  chanceCreationPassing  \\\ncount           1458.000000         1458.000000            1458.000000   \nmean              48.607362           48.490398              52.165295   \nstd                5.601170           10.896101              10.360793   \nmin               24.000000           20.000000              21.000000   \n25%               48.607362           40.000000              46.000000   \n50%               48.607362           50.000000              52.000000   \n75%               48.607362           55.000000              59.000000   \nmax               77.000000           80.000000              80.000000   \n\n       chanceCreationCrossing  chanceCreationShooting  defencePressure  \\\ncount             1458.000000             1458.000000      1458.000000   \nmean                53.731824               53.969136        46.017147   \nstd                 11.086796               10.327566        10.227225   \nmin                 20.000000               22.000000        23.000000   \n25%                 47.000000               48.000000        39.000000   \n50%                 53.000000               53.000000        45.000000   \n75%                 62.000000               61.000000        51.000000   \nmax                 80.000000               80.000000        72.000000   \n\n       defenceAggression  defenceTeamWidth    avg_shots  avg_corners  \ncount        1458.000000       1458.000000  1458.000000   997.000000  \nmean           49.251029         52.185871    10.579949    10.807166  \nstd             9.738028          9.574712     1.495291     1.916837  \nmin            24.000000         29.000000     3.000000     3.000000  \n25%            44.000000         47.000000     9.947368     9.700000  \n50%            48.000000         52.000000    10.579949    10.591837  \n75%            55.000000         58.000000    10.960000    11.526316  \nmax            72.000000         73.000000    16.000000    21.000000  "
  },
  "execution_count": 158,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

As we can see, there are a lot of Nan values on the avg_corners column. This
represents teams that did not have corners data on this dataset. Instead of
removing thoose rows, and give less input to our models we need again to do
**mean imputation** and deal with these values.

```{.python .input  n=159}
teams_stats['avg_corners'].hist();
```

```{.json .output n=159}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

We can see that most avg_corners values fall within the 8.5 - 12 range, so let's
fill in these entries with the average measured avg_corners

```{.python .input  n=160}
corners_avg_team_avg = teams_stats['avg_corners'].mean()
# mean imputation
teams_stats.loc[(teams_stats['avg_corners'].isnull()), 'avg_corners'] = corners_avg_team_avg
# showing new values
teams_stats.describe()
```

```{.json .output n=160}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>team_fifa_api_id</th>\n      <th>team_api_id</th>\n      <th>buildUpPlaySpeed</th>\n      <th>buildUpPlayDribbling</th>\n      <th>buildUpPlayPassing</th>\n      <th>chanceCreationPassing</th>\n      <th>chanceCreationCrossing</th>\n      <th>chanceCreationShooting</th>\n      <th>defencePressure</th>\n      <th>defenceAggression</th>\n      <th>defenceTeamWidth</th>\n      <th>avg_shots</th>\n      <th>avg_corners</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>729.500000</td>\n      <td>17706.982167</td>\n      <td>9995.727023</td>\n      <td>52.462277</td>\n      <td>48.607362</td>\n      <td>48.490398</td>\n      <td>52.165295</td>\n      <td>53.731824</td>\n      <td>53.969136</td>\n      <td>46.017147</td>\n      <td>49.251029</td>\n      <td>52.185871</td>\n      <td>10.579949</td>\n      <td>10.807166</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>421.032659</td>\n      <td>39179.857739</td>\n      <td>13264.869900</td>\n      <td>11.545869</td>\n      <td>5.601170</td>\n      <td>10.896101</td>\n      <td>10.360793</td>\n      <td>11.086796</td>\n      <td>10.327566</td>\n      <td>10.227225</td>\n      <td>9.738028</td>\n      <td>9.574712</td>\n      <td>1.495291</td>\n      <td>1.584839</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>1601.000000</td>\n      <td>20.000000</td>\n      <td>24.000000</td>\n      <td>20.000000</td>\n      <td>21.000000</td>\n      <td>20.000000</td>\n      <td>22.000000</td>\n      <td>23.000000</td>\n      <td>24.000000</td>\n      <td>29.000000</td>\n      <td>3.000000</td>\n      <td>3.000000</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>365.250000</td>\n      <td>110.000000</td>\n      <td>8457.750000</td>\n      <td>45.000000</td>\n      <td>48.607362</td>\n      <td>40.000000</td>\n      <td>46.000000</td>\n      <td>47.000000</td>\n      <td>48.000000</td>\n      <td>39.000000</td>\n      <td>44.000000</td>\n      <td>47.000000</td>\n      <td>9.947368</td>\n      <td>10.119718</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>729.500000</td>\n      <td>485.000000</td>\n      <td>8674.000000</td>\n      <td>52.000000</td>\n      <td>48.607362</td>\n      <td>50.000000</td>\n      <td>52.000000</td>\n      <td>53.000000</td>\n      <td>53.000000</td>\n      <td>45.000000</td>\n      <td>48.000000</td>\n      <td>52.000000</td>\n      <td>10.579949</td>\n      <td>10.807166</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>1093.750000</td>\n      <td>1900.000000</td>\n      <td>9904.000000</td>\n      <td>62.000000</td>\n      <td>48.607362</td>\n      <td>55.000000</td>\n      <td>59.000000</td>\n      <td>62.000000</td>\n      <td>61.000000</td>\n      <td>51.000000</td>\n      <td>55.000000</td>\n      <td>58.000000</td>\n      <td>10.960000</td>\n      <td>11.105263</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>1458.000000</td>\n      <td>112513.000000</td>\n      <td>274581.000000</td>\n      <td>80.000000</td>\n      <td>77.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>72.000000</td>\n      <td>72.000000</td>\n      <td>73.000000</td>\n      <td>16.000000</td>\n      <td>21.000000</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                id  team_fifa_api_id    team_api_id  buildUpPlaySpeed  \\\ncount  1458.000000       1458.000000    1458.000000       1458.000000   \nmean    729.500000      17706.982167    9995.727023         52.462277   \nstd     421.032659      39179.857739   13264.869900         11.545869   \nmin       1.000000          1.000000    1601.000000         20.000000   \n25%     365.250000        110.000000    8457.750000         45.000000   \n50%     729.500000        485.000000    8674.000000         52.000000   \n75%    1093.750000       1900.000000    9904.000000         62.000000   \nmax    1458.000000     112513.000000  274581.000000         80.000000   \n\n       buildUpPlayDribbling  buildUpPlayPassing  chanceCreationPassing  \\\ncount           1458.000000         1458.000000            1458.000000   \nmean              48.607362           48.490398              52.165295   \nstd                5.601170           10.896101              10.360793   \nmin               24.000000           20.000000              21.000000   \n25%               48.607362           40.000000              46.000000   \n50%               48.607362           50.000000              52.000000   \n75%               48.607362           55.000000              59.000000   \nmax               77.000000           80.000000              80.000000   \n\n       chanceCreationCrossing  chanceCreationShooting  defencePressure  \\\ncount             1458.000000             1458.000000      1458.000000   \nmean                53.731824               53.969136        46.017147   \nstd                 11.086796               10.327566        10.227225   \nmin                 20.000000               22.000000        23.000000   \n25%                 47.000000               48.000000        39.000000   \n50%                 53.000000               53.000000        45.000000   \n75%                 62.000000               61.000000        51.000000   \nmax                 80.000000               80.000000        72.000000   \n\n       defenceAggression  defenceTeamWidth    avg_shots  avg_corners  \ncount        1458.000000       1458.000000  1458.000000  1458.000000  \nmean           49.251029         52.185871    10.579949    10.807166  \nstd             9.738028          9.574712     1.495291     1.584839  \nmin            24.000000         29.000000     3.000000     3.000000  \n25%            44.000000         47.000000     9.947368    10.119718  \n50%            48.000000         52.000000    10.579949    10.807166  \n75%            55.000000         58.000000    10.960000    11.105263  \nmax            72.000000         73.000000    16.000000    21.000000  "
  },
  "execution_count": 160,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Having done the **mean imputation** for team_attributes we can see that there
are no longer missing values for the avg_corners.

```{.python .input  n=161}
teams_stats.loc[(teams_stats['avg_corners'].isnull())]
```

```{.json .output n=161}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>team_fifa_api_id</th>\n      <th>team_api_id</th>\n      <th>date</th>\n      <th>buildUpPlaySpeed</th>\n      <th>buildUpPlayDribbling</th>\n      <th>buildUpPlayPassing</th>\n      <th>chanceCreationPassing</th>\n      <th>chanceCreationCrossing</th>\n      <th>chanceCreationShooting</th>\n      <th>defencePressure</th>\n      <th>defenceAggression</th>\n      <th>defenceTeamWidth</th>\n      <th>avg_shots</th>\n      <th>avg_corners</th>\n    </tr>\n  </thead>\n  <tbody>\n  </tbody>\n</table>\n</div>",
   "text/plain": "Empty DataFrame\nColumns: [id, team_fifa_api_id, team_api_id, date, buildUpPlaySpeed, buildUpPlayDribbling, buildUpPlayPassing, chanceCreationPassing, chanceCreationCrossing, chanceCreationShooting, defencePressure, defenceAggression, defenceTeamWidth, avg_shots, avg_corners]\nIndex: []"
  },
  "execution_count": 161,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### Team Stats - Crosses
The final feature to be added is the crosses data. Normally the more dominant
team has more crosses because it creates more opportunity of goals during a
match. After merging all the data we need to watch for missing rows on the new
added column.

```{.python .input  n=162}
crosses_data = pd.read_csv("../dataset/cross_detail.csv")

crosses = crosses_data[['match_id', 'team']]
total_crosses = crosses["team"].value_counts()
total_matches = crosses.drop_duplicates(['match_id', 'team'])["team"].value_counts()

for index, n_crosses in total_crosses.iteritems():
    n_matches = total_matches[index]
    avg_crosses = n_crosses / n_matches
    teams_stats.loc[teams_stats['team_api_id'] == index, 'avg_crosses'] = avg_crosses

teams_stats.describe()
```

```{.json .output n=162}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "C:\\Python\\Python38\\lib\\site-packages\\IPython\\core\\interactiveshell.py:3062: DtypeWarning: Columns (8,17) have mixed types.Specify dtype option on import or set low_memory=False.\n  has_raised = await self.run_ast_nodes(code_ast.body, cell_name,\n"
 },
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>team_fifa_api_id</th>\n      <th>team_api_id</th>\n      <th>buildUpPlaySpeed</th>\n      <th>buildUpPlayDribbling</th>\n      <th>buildUpPlayPassing</th>\n      <th>chanceCreationPassing</th>\n      <th>chanceCreationCrossing</th>\n      <th>chanceCreationShooting</th>\n      <th>defencePressure</th>\n      <th>defenceAggression</th>\n      <th>defenceTeamWidth</th>\n      <th>avg_shots</th>\n      <th>avg_corners</th>\n      <th>avg_crosses</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>997.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>729.500000</td>\n      <td>17706.982167</td>\n      <td>9995.727023</td>\n      <td>52.462277</td>\n      <td>48.607362</td>\n      <td>48.490398</td>\n      <td>52.165295</td>\n      <td>53.731824</td>\n      <td>53.969136</td>\n      <td>46.017147</td>\n      <td>49.251029</td>\n      <td>52.185871</td>\n      <td>10.579949</td>\n      <td>10.807166</td>\n      <td>15.888237</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>421.032659</td>\n      <td>39179.857739</td>\n      <td>13264.869900</td>\n      <td>11.545869</td>\n      <td>5.601170</td>\n      <td>10.896101</td>\n      <td>10.360793</td>\n      <td>11.086796</td>\n      <td>10.327566</td>\n      <td>10.227225</td>\n      <td>9.738028</td>\n      <td>9.574712</td>\n      <td>1.495291</td>\n      <td>1.584839</td>\n      <td>2.815376</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>1601.000000</td>\n      <td>20.000000</td>\n      <td>24.000000</td>\n      <td>20.000000</td>\n      <td>21.000000</td>\n      <td>20.000000</td>\n      <td>22.000000</td>\n      <td>23.000000</td>\n      <td>24.000000</td>\n      <td>29.000000</td>\n      <td>3.000000</td>\n      <td>3.000000</td>\n      <td>1.500000</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>365.250000</td>\n      <td>110.000000</td>\n      <td>8457.750000</td>\n      <td>45.000000</td>\n      <td>48.607362</td>\n      <td>40.000000</td>\n      <td>46.000000</td>\n      <td>47.000000</td>\n      <td>48.000000</td>\n      <td>39.000000</td>\n      <td>44.000000</td>\n      <td>47.000000</td>\n      <td>9.947368</td>\n      <td>10.119718</td>\n      <td>14.358974</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>729.500000</td>\n      <td>485.000000</td>\n      <td>8674.000000</td>\n      <td>52.000000</td>\n      <td>48.607362</td>\n      <td>50.000000</td>\n      <td>52.000000</td>\n      <td>53.000000</td>\n      <td>53.000000</td>\n      <td>45.000000</td>\n      <td>48.000000</td>\n      <td>52.000000</td>\n      <td>10.579949</td>\n      <td>10.807166</td>\n      <td>15.750000</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>1093.750000</td>\n      <td>1900.000000</td>\n      <td>9904.000000</td>\n      <td>62.000000</td>\n      <td>48.607362</td>\n      <td>55.000000</td>\n      <td>59.000000</td>\n      <td>62.000000</td>\n      <td>61.000000</td>\n      <td>51.000000</td>\n      <td>55.000000</td>\n      <td>58.000000</td>\n      <td>10.960000</td>\n      <td>11.105263</td>\n      <td>17.782895</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>1458.000000</td>\n      <td>112513.000000</td>\n      <td>274581.000000</td>\n      <td>80.000000</td>\n      <td>77.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>72.000000</td>\n      <td>72.000000</td>\n      <td>73.000000</td>\n      <td>16.000000</td>\n      <td>21.000000</td>\n      <td>27.000000</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                id  team_fifa_api_id    team_api_id  buildUpPlaySpeed  \\\ncount  1458.000000       1458.000000    1458.000000       1458.000000   \nmean    729.500000      17706.982167    9995.727023         52.462277   \nstd     421.032659      39179.857739   13264.869900         11.545869   \nmin       1.000000          1.000000    1601.000000         20.000000   \n25%     365.250000        110.000000    8457.750000         45.000000   \n50%     729.500000        485.000000    8674.000000         52.000000   \n75%    1093.750000       1900.000000    9904.000000         62.000000   \nmax    1458.000000     112513.000000  274581.000000         80.000000   \n\n       buildUpPlayDribbling  buildUpPlayPassing  chanceCreationPassing  \\\ncount           1458.000000         1458.000000            1458.000000   \nmean              48.607362           48.490398              52.165295   \nstd                5.601170           10.896101              10.360793   \nmin               24.000000           20.000000              21.000000   \n25%               48.607362           40.000000              46.000000   \n50%               48.607362           50.000000              52.000000   \n75%               48.607362           55.000000              59.000000   \nmax               77.000000           80.000000              80.000000   \n\n       chanceCreationCrossing  chanceCreationShooting  defencePressure  \\\ncount             1458.000000             1458.000000      1458.000000   \nmean                53.731824               53.969136        46.017147   \nstd                 11.086796               10.327566        10.227225   \nmin                 20.000000               22.000000        23.000000   \n25%                 47.000000               48.000000        39.000000   \n50%                 53.000000               53.000000        45.000000   \n75%                 62.000000               61.000000        51.000000   \nmax                 80.000000               80.000000        72.000000   \n\n       defenceAggression  defenceTeamWidth    avg_shots  avg_corners  \\\ncount        1458.000000       1458.000000  1458.000000  1458.000000   \nmean           49.251029         52.185871    10.579949    10.807166   \nstd             9.738028          9.574712     1.495291     1.584839   \nmin            24.000000         29.000000     3.000000     3.000000   \n25%            44.000000         47.000000     9.947368    10.119718   \n50%            48.000000         52.000000    10.579949    10.807166   \n75%            55.000000         58.000000    10.960000    11.105263   \nmax            72.000000         73.000000    16.000000    21.000000   \n\n       avg_crosses  \ncount   997.000000  \nmean     15.888237  \nstd       2.815376  \nmin       1.500000  \n25%      14.358974  \n50%      15.750000  \n75%      17.782895  \nmax      27.000000  "
  },
  "execution_count": 162,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

As we can see, there are a lot of Nan values on the avg_crosses column. This
represents teams that did not have crosses data on this dataset. Instead of
removing thoose rows, and give less input to our models we need again to do
**mean imputation** and deal with these values

```{.python .input  n=163}
teams_stats['avg_crosses'].hist();
```

```{.json .output n=163}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

We can see that most avg_crosses values fall within the 12.5 - 17.5 range, so
let's fill in these entries with the average measured avg_corners

```{.python .input  n=164}
crosses_avg_team_avg = teams_stats['avg_crosses'].mean()
# mean imputation
teams_stats.loc[(teams_stats['avg_crosses'].isnull()), 'avg_crosses'] = crosses_avg_team_avg
# showing new values
teams_stats.describe()
```

```{.json .output n=164}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>team_fifa_api_id</th>\n      <th>team_api_id</th>\n      <th>buildUpPlaySpeed</th>\n      <th>buildUpPlayDribbling</th>\n      <th>buildUpPlayPassing</th>\n      <th>chanceCreationPassing</th>\n      <th>chanceCreationCrossing</th>\n      <th>chanceCreationShooting</th>\n      <th>defencePressure</th>\n      <th>defenceAggression</th>\n      <th>defenceTeamWidth</th>\n      <th>avg_shots</th>\n      <th>avg_corners</th>\n      <th>avg_crosses</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n      <td>1458.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>729.500000</td>\n      <td>17706.982167</td>\n      <td>9995.727023</td>\n      <td>52.462277</td>\n      <td>48.607362</td>\n      <td>48.490398</td>\n      <td>52.165295</td>\n      <td>53.731824</td>\n      <td>53.969136</td>\n      <td>46.017147</td>\n      <td>49.251029</td>\n      <td>52.185871</td>\n      <td>10.579949</td>\n      <td>10.807166</td>\n      <td>15.888237</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>421.032659</td>\n      <td>39179.857739</td>\n      <td>13264.869900</td>\n      <td>11.545869</td>\n      <td>5.601170</td>\n      <td>10.896101</td>\n      <td>10.360793</td>\n      <td>11.086796</td>\n      <td>10.327566</td>\n      <td>10.227225</td>\n      <td>9.738028</td>\n      <td>9.574712</td>\n      <td>1.495291</td>\n      <td>1.584839</td>\n      <td>2.327750</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>1601.000000</td>\n      <td>20.000000</td>\n      <td>24.000000</td>\n      <td>20.000000</td>\n      <td>21.000000</td>\n      <td>20.000000</td>\n      <td>22.000000</td>\n      <td>23.000000</td>\n      <td>24.000000</td>\n      <td>29.000000</td>\n      <td>3.000000</td>\n      <td>3.000000</td>\n      <td>1.500000</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>365.250000</td>\n      <td>110.000000</td>\n      <td>8457.750000</td>\n      <td>45.000000</td>\n      <td>48.607362</td>\n      <td>40.000000</td>\n      <td>46.000000</td>\n      <td>47.000000</td>\n      <td>48.000000</td>\n      <td>39.000000</td>\n      <td>44.000000</td>\n      <td>47.000000</td>\n      <td>9.947368</td>\n      <td>10.119718</td>\n      <td>14.971448</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>729.500000</td>\n      <td>485.000000</td>\n      <td>8674.000000</td>\n      <td>52.000000</td>\n      <td>48.607362</td>\n      <td>50.000000</td>\n      <td>52.000000</td>\n      <td>53.000000</td>\n      <td>53.000000</td>\n      <td>45.000000</td>\n      <td>48.000000</td>\n      <td>52.000000</td>\n      <td>10.579949</td>\n      <td>10.807166</td>\n      <td>15.888237</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>1093.750000</td>\n      <td>1900.000000</td>\n      <td>9904.000000</td>\n      <td>62.000000</td>\n      <td>48.607362</td>\n      <td>55.000000</td>\n      <td>59.000000</td>\n      <td>62.000000</td>\n      <td>61.000000</td>\n      <td>51.000000</td>\n      <td>55.000000</td>\n      <td>58.000000</td>\n      <td>10.960000</td>\n      <td>11.105263</td>\n      <td>16.671053</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>1458.000000</td>\n      <td>112513.000000</td>\n      <td>274581.000000</td>\n      <td>80.000000</td>\n      <td>77.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>80.000000</td>\n      <td>72.000000</td>\n      <td>72.000000</td>\n      <td>73.000000</td>\n      <td>16.000000</td>\n      <td>21.000000</td>\n      <td>27.000000</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                id  team_fifa_api_id    team_api_id  buildUpPlaySpeed  \\\ncount  1458.000000       1458.000000    1458.000000       1458.000000   \nmean    729.500000      17706.982167    9995.727023         52.462277   \nstd     421.032659      39179.857739   13264.869900         11.545869   \nmin       1.000000          1.000000    1601.000000         20.000000   \n25%     365.250000        110.000000    8457.750000         45.000000   \n50%     729.500000        485.000000    8674.000000         52.000000   \n75%    1093.750000       1900.000000    9904.000000         62.000000   \nmax    1458.000000     112513.000000  274581.000000         80.000000   \n\n       buildUpPlayDribbling  buildUpPlayPassing  chanceCreationPassing  \\\ncount           1458.000000         1458.000000            1458.000000   \nmean              48.607362           48.490398              52.165295   \nstd                5.601170           10.896101              10.360793   \nmin               24.000000           20.000000              21.000000   \n25%               48.607362           40.000000              46.000000   \n50%               48.607362           50.000000              52.000000   \n75%               48.607362           55.000000              59.000000   \nmax               77.000000           80.000000              80.000000   \n\n       chanceCreationCrossing  chanceCreationShooting  defencePressure  \\\ncount             1458.000000             1458.000000      1458.000000   \nmean                53.731824               53.969136        46.017147   \nstd                 11.086796               10.327566        10.227225   \nmin                 20.000000               22.000000        23.000000   \n25%                 47.000000               48.000000        39.000000   \n50%                 53.000000               53.000000        45.000000   \n75%                 62.000000               61.000000        51.000000   \nmax                 80.000000               80.000000        72.000000   \n\n       defenceAggression  defenceTeamWidth    avg_shots  avg_corners  \\\ncount        1458.000000       1458.000000  1458.000000  1458.000000   \nmean           49.251029         52.185871    10.579949    10.807166   \nstd             9.738028          9.574712     1.495291     1.584839   \nmin            24.000000         29.000000     3.000000     3.000000   \n25%            44.000000         47.000000     9.947368    10.119718   \n50%            48.000000         52.000000    10.579949    10.807166   \n75%            55.000000         58.000000    10.960000    11.105263   \nmax            72.000000         73.000000    16.000000    21.000000   \n\n       avg_crosses  \ncount  1458.000000  \nmean     15.888237  \nstd       2.327750  \nmin       1.500000  \n25%      14.971448  \n50%      15.888237  \n75%      16.671053  \nmax      27.000000  "
  },
  "execution_count": 164,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Having done the **mean imputation** for team_attributes we can see that there
are no longer missing values for the avg_crosses.

```{.python .input  n=165}
teams_stats.loc[(teams_stats['avg_crosses'].isnull())]
```

```{.json .output n=165}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>team_fifa_api_id</th>\n      <th>team_api_id</th>\n      <th>date</th>\n      <th>buildUpPlaySpeed</th>\n      <th>buildUpPlayDribbling</th>\n      <th>buildUpPlayPassing</th>\n      <th>chanceCreationPassing</th>\n      <th>chanceCreationCrossing</th>\n      <th>chanceCreationShooting</th>\n      <th>defencePressure</th>\n      <th>defenceAggression</th>\n      <th>defenceTeamWidth</th>\n      <th>avg_shots</th>\n      <th>avg_corners</th>\n      <th>avg_crosses</th>\n    </tr>\n  </thead>\n  <tbody>\n  </tbody>\n</table>\n</div>",
   "text/plain": "Empty DataFrame\nColumns: [id, team_fifa_api_id, team_api_id, date, buildUpPlaySpeed, buildUpPlayDribbling, buildUpPlayPassing, chanceCreationPassing, chanceCreationCrossing, chanceCreationShooting, defencePressure, defenceAggression, defenceTeamWidth, avg_shots, avg_corners, avg_crosses]\nIndex: []"
  },
  "execution_count": 165,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### FIFA data
We will now gather the fifa data regarding the overrall rating of the teams.
This will create some columns that will include the overall ratings of the
players that belong to a team. This way we can more easily compare the value of
each team.

```{.python .input  n=166}
def get_fifa_stats(match, player_stats):
    ''' Aggregates fifa stats for a given match. '''    
    
    #Define variables
    match_id =  match.match_api_id
    date = match['date']
    players = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11", "away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]
    player_stats_new = pd.DataFrame()
    names = []
    
    #Loop through all players
    for player in players:   
            
        #Get player ID
        player_id = match[player]
        
        #Get player stats 
        stats = player_stats[player_stats.player_api_id == player_id]
            
        #Identify current stats       
        current_stats = stats[stats.date < date].sort_values(by = 'date', ascending = False)[:1]
        
        if np.isnan(player_id) == True:
            overall_rating = pd.Series(0)
        else:
            current_stats.reset_index(inplace = True, drop = True)
            overall_rating = pd.Series(current_stats.loc[0, "overall_rating"])

        #Rename stat
        name = "{}_overall_rating".format(player)
        names.append(name)
            
        #Aggregate stats
        player_stats_new = pd.concat([player_stats_new, overall_rating], axis = 1)
    
    player_stats_new.columns = names        
    player_stats_new['match_api_id'] = match_id

    player_stats_new.reset_index(inplace = True, drop = True)
    
    #Return player stats    
    return player_stats_new.iloc[0]    


def get_fifa_data(matches, player_stats, path = None, data_exists = False):
    ''' Gets fifa data for all matches. '''  
    
    #Check if fifa data already exists
    if data_exists == True:
        fifa_data = pd.read_pickle(path)
        
    else:
        print("Collecting fifa data for each match...")       
        start = time()
        
        #Apply get_fifa_stats for each match
        fifa_data = matches.apply(lambda x :get_fifa_stats(x, player_stats), axis = 1)
        
        end = time()    
        print("Fifa data collected in {:.1f} minutes".format((end - start)/60))
    
    #Return fifa_data
    return fifa_data
```

```{.python .input  n=167}
fifa_data = get_fifa_data(viable_matches, player_attributes, None, data_exists = False)
fifa_data.describe()
```

```{.json .output n=167}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Collecting fifa data for each match...\nFifa data collected in 8.5 minutes\n"
 },
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>home_player_1_overall_rating</th>\n      <th>home_player_2_overall_rating</th>\n      <th>home_player_3_overall_rating</th>\n      <th>home_player_4_overall_rating</th>\n      <th>home_player_5_overall_rating</th>\n      <th>home_player_6_overall_rating</th>\n      <th>home_player_7_overall_rating</th>\n      <th>home_player_8_overall_rating</th>\n      <th>home_player_9_overall_rating</th>\n      <th>home_player_10_overall_rating</th>\n      <th>...</th>\n      <th>away_player_3_overall_rating</th>\n      <th>away_player_4_overall_rating</th>\n      <th>away_player_5_overall_rating</th>\n      <th>away_player_6_overall_rating</th>\n      <th>away_player_7_overall_rating</th>\n      <th>away_player_8_overall_rating</th>\n      <th>away_player_9_overall_rating</th>\n      <th>away_player_10_overall_rating</th>\n      <th>away_player_11_overall_rating</th>\n      <th>match_api_id</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>...</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2505.000000</td>\n      <td>2.505000e+03</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>76.027944</td>\n      <td>72.903792</td>\n      <td>74.638723</td>\n      <td>74.544112</td>\n      <td>72.603992</td>\n      <td>74.267066</td>\n      <td>74.444311</td>\n      <td>74.532136</td>\n      <td>75.213573</td>\n      <td>75.596008</td>\n      <td>...</td>\n      <td>74.584431</td>\n      <td>74.300599</td>\n      <td>72.453094</td>\n      <td>74.118962</td>\n      <td>74.238323</td>\n      <td>74.347305</td>\n      <td>75.067066</td>\n      <td>75.324551</td>\n      <td>75.674651</td>\n      <td>1.273188e+06</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>5.534673</td>\n      <td>4.996397</td>\n      <td>5.050940</td>\n      <td>5.255431</td>\n      <td>5.177256</td>\n      <td>5.303050</td>\n      <td>5.241549</td>\n      <td>5.252022</td>\n      <td>5.392455</td>\n      <td>5.695381</td>\n      <td>...</td>\n      <td>4.971982</td>\n      <td>5.357903</td>\n      <td>5.081081</td>\n      <td>5.237664</td>\n      <td>5.205855</td>\n      <td>5.287646</td>\n      <td>5.526032</td>\n      <td>5.744533</td>\n      <td>5.467002</td>\n      <td>4.837662e+05</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>43.000000</td>\n      <td>53.000000</td>\n      <td>56.000000</td>\n      <td>54.000000</td>\n      <td>53.000000</td>\n      <td>48.000000</td>\n      <td>45.000000</td>\n      <td>51.000000</td>\n      <td>51.000000</td>\n      <td>55.000000</td>\n      <td>...</td>\n      <td>54.000000</td>\n      <td>51.000000</td>\n      <td>52.000000</td>\n      <td>46.000000</td>\n      <td>50.000000</td>\n      <td>53.000000</td>\n      <td>51.000000</td>\n      <td>54.000000</td>\n      <td>55.000000</td>\n      <td>4.890430e+05</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>73.000000</td>\n      <td>70.000000</td>\n      <td>72.000000</td>\n      <td>71.000000</td>\n      <td>69.000000</td>\n      <td>71.000000</td>\n      <td>71.000000</td>\n      <td>71.000000</td>\n      <td>72.000000</td>\n      <td>72.000000</td>\n      <td>...</td>\n      <td>71.000000</td>\n      <td>71.000000</td>\n      <td>69.000000</td>\n      <td>71.000000</td>\n      <td>71.000000</td>\n      <td>71.000000</td>\n      <td>72.000000</td>\n      <td>72.000000</td>\n      <td>72.000000</td>\n      <td>8.572140e+05</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>76.000000</td>\n      <td>73.000000</td>\n      <td>75.000000</td>\n      <td>75.000000</td>\n      <td>73.000000</td>\n      <td>74.000000</td>\n      <td>74.000000</td>\n      <td>75.000000</td>\n      <td>75.000000</td>\n      <td>75.000000</td>\n      <td>...</td>\n      <td>75.000000</td>\n      <td>74.000000</td>\n      <td>73.000000</td>\n      <td>74.000000</td>\n      <td>74.000000</td>\n      <td>74.000000</td>\n      <td>75.000000</td>\n      <td>75.000000</td>\n      <td>75.000000</td>\n      <td>1.239632e+06</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>80.000000</td>\n      <td>76.000000</td>\n      <td>78.000000</td>\n      <td>78.000000</td>\n      <td>76.000000</td>\n      <td>78.000000</td>\n      <td>78.000000</td>\n      <td>78.000000</td>\n      <td>79.000000</td>\n      <td>79.000000</td>\n      <td>...</td>\n      <td>78.000000</td>\n      <td>78.000000</td>\n      <td>76.000000</td>\n      <td>77.000000</td>\n      <td>77.000000</td>\n      <td>78.000000</td>\n      <td>78.000000</td>\n      <td>79.000000</td>\n      <td>79.000000</td>\n      <td>1.724026e+06</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>91.000000</td>\n      <td>88.000000</td>\n      <td>89.000000</td>\n      <td>89.000000</td>\n      <td>88.000000</td>\n      <td>91.000000</td>\n      <td>90.000000</td>\n      <td>91.000000</td>\n      <td>94.000000</td>\n      <td>94.000000</td>\n      <td>...</td>\n      <td>89.000000</td>\n      <td>90.000000</td>\n      <td>88.000000</td>\n      <td>92.000000</td>\n      <td>91.000000</td>\n      <td>91.000000</td>\n      <td>94.000000</td>\n      <td>94.000000</td>\n      <td>93.000000</td>\n      <td>2.060644e+06</td>\n    </tr>\n  </tbody>\n</table>\n<p>8 rows \u00d7 23 columns</p>\n</div>",
   "text/plain": "       home_player_1_overall_rating  home_player_2_overall_rating  \\\ncount                   2505.000000                   2505.000000   \nmean                      76.027944                     72.903792   \nstd                        5.534673                      4.996397   \nmin                       43.000000                     53.000000   \n25%                       73.000000                     70.000000   \n50%                       76.000000                     73.000000   \n75%                       80.000000                     76.000000   \nmax                       91.000000                     88.000000   \n\n       home_player_3_overall_rating  home_player_4_overall_rating  \\\ncount                   2505.000000                   2505.000000   \nmean                      74.638723                     74.544112   \nstd                        5.050940                      5.255431   \nmin                       56.000000                     54.000000   \n25%                       72.000000                     71.000000   \n50%                       75.000000                     75.000000   \n75%                       78.000000                     78.000000   \nmax                       89.000000                     89.000000   \n\n       home_player_5_overall_rating  home_player_6_overall_rating  \\\ncount                   2505.000000                   2505.000000   \nmean                      72.603992                     74.267066   \nstd                        5.177256                      5.303050   \nmin                       53.000000                     48.000000   \n25%                       69.000000                     71.000000   \n50%                       73.000000                     74.000000   \n75%                       76.000000                     78.000000   \nmax                       88.000000                     91.000000   \n\n       home_player_7_overall_rating  home_player_8_overall_rating  \\\ncount                   2505.000000                   2505.000000   \nmean                      74.444311                     74.532136   \nstd                        5.241549                      5.252022   \nmin                       45.000000                     51.000000   \n25%                       71.000000                     71.000000   \n50%                       74.000000                     75.000000   \n75%                       78.000000                     78.000000   \nmax                       90.000000                     91.000000   \n\n       home_player_9_overall_rating  home_player_10_overall_rating  ...  \\\ncount                   2505.000000                    2505.000000  ...   \nmean                      75.213573                      75.596008  ...   \nstd                        5.392455                       5.695381  ...   \nmin                       51.000000                      55.000000  ...   \n25%                       72.000000                      72.000000  ...   \n50%                       75.000000                      75.000000  ...   \n75%                       79.000000                      79.000000  ...   \nmax                       94.000000                      94.000000  ...   \n\n       away_player_3_overall_rating  away_player_4_overall_rating  \\\ncount                   2505.000000                   2505.000000   \nmean                      74.584431                     74.300599   \nstd                        4.971982                      5.357903   \nmin                       54.000000                     51.000000   \n25%                       71.000000                     71.000000   \n50%                       75.000000                     74.000000   \n75%                       78.000000                     78.000000   \nmax                       89.000000                     90.000000   \n\n       away_player_5_overall_rating  away_player_6_overall_rating  \\\ncount                   2505.000000                   2505.000000   \nmean                      72.453094                     74.118962   \nstd                        5.081081                      5.237664   \nmin                       52.000000                     46.000000   \n25%                       69.000000                     71.000000   \n50%                       73.000000                     74.000000   \n75%                       76.000000                     77.000000   \nmax                       88.000000                     92.000000   \n\n       away_player_7_overall_rating  away_player_8_overall_rating  \\\ncount                   2505.000000                   2505.000000   \nmean                      74.238323                     74.347305   \nstd                        5.205855                      5.287646   \nmin                       50.000000                     53.000000   \n25%                       71.000000                     71.000000   \n50%                       74.000000                     74.000000   \n75%                       77.000000                     78.000000   \nmax                       91.000000                     91.000000   \n\n       away_player_9_overall_rating  away_player_10_overall_rating  \\\ncount                   2505.000000                    2505.000000   \nmean                      75.067066                      75.324551   \nstd                        5.526032                       5.744533   \nmin                       51.000000                      54.000000   \n25%                       72.000000                      72.000000   \n50%                       75.000000                      75.000000   \n75%                       78.000000                      79.000000   \nmax                       94.000000                      94.000000   \n\n       away_player_11_overall_rating  match_api_id  \ncount                    2505.000000  2.505000e+03  \nmean                       75.674651  1.273188e+06  \nstd                         5.467002  4.837662e+05  \nmin                        55.000000  4.890430e+05  \n25%                        72.000000  8.572140e+05  \n50%                        75.000000  1.239632e+06  \n75%                        79.000000  1.724026e+06  \nmax                        93.000000  2.060644e+06  \n\n[8 rows x 23 columns]"
  },
  "execution_count": 167,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### Joining all features
In this instance we need to join all features, select the input we will pass to
our models and drop the column regarding the outcome label.
To improve the overall measures of the supervised learning models we need to
normalize our features before training our models.

```{.python .input  n=168}
# Creates features and labels based on the provided data
viables = get_features(viable_matches, teams_stats, fifa_data, 10, False)
inputs = viables.drop('match_api_id', axis=1)
outcomes = inputs.loc[:, 'outcome']
# all features except outcomes
features = inputs.drop('outcome', axis=1)
features.iloc[:,:] = Normalizer(norm='l1').fit_transform(features)
features.head()
```

## Step 2: Classification & Results Interpretation
[go back to the top](#Table-of-contents)

### Training and Evaluating Models

We used **K-Fold Cross validation**. The idea is that we split the dataset,
after processing it, in k bins of equal size to better estimate the skill of a
model on unseen data. That is, to use a limited sample in order to estimate how
the model is expected to perform in general when used to make predictions on
data not used during the training of the model. It results in a less biased
estimate of the model skill in comparision to the typical train_test_split.

```{.python .input  n=178}
def train_predict(clf, data, outcomes):
    y_predict = train_model(clf, data, outcomes)
    predict_metrics(outcomes, y_predict)

def train_predict_nn(clf, data, outcomes):
    le = LabelEncoder()
    y_outcomes = le.fit_transform(outcomes)
    y_outcomes = np_utils.to_categorical(y_outcomes)
    
    y_predict = train_model_nn(clf, data, y_outcomes)
    
    y_predict_reverse = [np.argmax(y, axis=None, out=None) for y in y_predict]
    y_predict_decoded = le.inverse_transform(y_predict_reverse)
    predict_metrics(outcomes, y_predict_decoded)

def train_model(clf, data, labels):
    kf = KFold(n_splits=5)
    predictions = []
    for train, test in kf.split(data):
        X_train, X_test = data[data.index.isin(train)], data[data.index.isin(test)]
        y_train, y_test = labels[data.index.isin(train)], labels[data.index.isin(test)]
        clf.fit(X_train, y_train)
        predictions.append(clf.predict(X_test))
        
    y_predict = predictions[0]
    y_predict = np.append(y_predict, predictions[1], axis=0)
    y_predict = np.append(y_predict, predictions[2], axis=0)
    y_predict = np.append(y_predict, predictions[3], axis=0)
    y_predict = np.append(y_predict, predictions[4], axis=0)
    
    return y_predict

def train_model_nn(clf, data, labels):
    kf = KFold(n_splits=5, shuffle=False)
    predictions = []
    for train, test in kf.split(data):
        X_train, X_test = data[data.index.isin(train)], data[data.index.isin(test)]
        y_train, y_test = labels[data.index.isin(train)], labels[data.index.isin(test)]
        clf.fit(X_train, y_train, epochs=20, verbose=0)
        predictions.append(clf.predict(X_test))
        
    y_predict = predictions[0]
    y_predict = np.append(y_predict, predictions[1], axis=0)
    y_predict = np.append(y_predict, predictions[2], axis=0)
    y_predict = np.append(y_predict, predictions[3], axis=0)
    y_predict = np.append(y_predict, predictions[4], axis=0)
        
    return y_predict

def predict_metrics(y_test, y_predict):
    ls = ['Win', 'Draw', 'Defeat']
    
    from sklearn import metrics
    cm = metrics.confusion_matrix(y_test, y_predict, ls)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ls)
    disp.plot(include_values=True, values_format='d')
    plt.show()
    
    print(metrics.classification_report(y_test, y_predict, target_names=ls))
    print("\n\nAccuracy: ", metrics.accuracy_score(y_test, y_predict))
    print("Recall: ", metrics.recall_score(y_test, y_predict, average='macro'))
    print("Precision: ", metrics.precision_score(y_test, y_predict, average='macro'))
    print("F1 Score: ", metrics.f1_score(y_test, y_predict, average='macro'))
```

### The basis

* The **accuracy** measure is great for balanced classes, but because this is
not the case we can't do much with it.

* **Precision-Recall** is a useful measure of success of prediction when the
classes are very imbalanced.

* **Precision** is a measure of the ability of a classification model to
identify only the relevant data points, ie, answers the question: what portion
of **predicted Positives** is truly Positive?

* **Recall** is a measure of the ability of a model to find all the relevant
cases within a dataset, ie, answers the question what portion of **actual
Positives** is correctly classified?

* **F1-Score** is a combination of both recall and precision. Because precision
and recall are often in tension, i.e, improving one will lead to a reduction in
the other, this way the f1-score is a great measure also.

### KNN

```{.python .input  n=171}
clf = KNeighborsClassifier(n_neighbors=100)
train_predict(clf, features, outcomes)
```

```{.json .output n=171}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 2 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "              precision    recall  f1-score   support\n\n         Win       0.51      0.24      0.33       707\n        Draw       0.42      0.02      0.04       637\n      Defeat       0.50      0.91      0.64      1161\n\n    accuracy                           0.50      2505\n   macro avg       0.48      0.39      0.34      2505\nweighted avg       0.48      0.50      0.40      2505\n\n\n\nAccuracy:  0.49820359281437127\nRecall:  0.39327758119903117\nPrecision:  0.4757010700244369\nF1 Score:  0.3380411900771032\n"
 }
]
```

### Decision Tree

```{.python .input  n=172}
clf = DecisionTreeClassifier(random_state=0, criterion='entropy', splitter='random', max_depth=5)
train_predict(clf, features, outcomes)
```

```{.json .output n=172}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 2 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "              precision    recall  f1-score   support\n\n         Win       0.47      0.39      0.42       707\n        Draw       0.29      0.06      0.09       637\n      Defeat       0.53      0.82      0.65      1161\n\n    accuracy                           0.50      2505\n   macro avg       0.43      0.42      0.39      2505\nweighted avg       0.45      0.50      0.44      2505\n\n\n\nAccuracy:  0.5045908183632735\nRecall:  0.42174009343948926\nPrecision:  0.4297484035759898\nF1 Score:  0.3879276534662697\n"
 }
]
```

### SVC

```{.python .input  n=173}
clf = SVC(coef0=5, kernel='poly')
train_predict(clf, features, outcomes)
```

```{.json .output n=173}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 2 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "              precision    recall  f1-score   support\n\n         Win       0.49      0.34      0.40       707\n        Draw       0.00      0.00      0.00       637\n      Defeat       0.51      0.89      0.65      1161\n\n    accuracy                           0.51      2505\n   macro avg       0.33      0.41      0.35      2505\nweighted avg       0.38      0.51      0.41      2505\n\n\n\nAccuracy:  0.5077844311377245\nRecall:  0.4088973681421298\nPrecision:  0.33495370370370364\nF1 Score:  0.3496811820459244\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "C:\\Python\\Python38\\lib\\site-packages\\sklearn\\metrics\\_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n  _warn_prf(average, modifier, msg_start, len(result))\nC:\\Python\\Python38\\lib\\site-packages\\sklearn\\metrics\\_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n  _warn_prf(average, modifier, msg_start, len(result))\n"
 }
]
```

### Naive Bayes

```{.python .input  n=174}
clf = GaussianNB(var_smoothing=1.1)
train_predict(clf, features, outcomes)
```

```{.json .output n=174}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 2 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "              precision    recall  f1-score   support\n\n         Win       0.46      0.48      0.47       707\n        Draw       0.37      0.03      0.05       637\n      Defeat       0.54      0.81      0.65      1161\n\n    accuracy                           0.52      2505\n   macro avg       0.46      0.44      0.39      2505\nweighted avg       0.48      0.52      0.45      2505\n\n\n\nAccuracy:  0.5157684630738523\nRecall:  0.436670706771471\nPrecision:  0.4585768924708331\nF1 Score:  0.38824900531845924\n"
 }
]
```

### Gradient Boosting

```{.python .input  n=175}
clf = XGBClassifier(max_depth=20)
train_predict(clf, features, outcomes)
```

```{.json .output n=175}
[
 {
  "data": {
   "image/png": "iVBORw0KGgoAAAANSUhEUgAAAVQAAAEICAYAAAAA3gw5AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAgAElEQVR4nO3deXxU1f3/8dc7KzshBBBZBGRRcEFFFLe60KK2X7ULSmv7RWuLtmirVSu2fqu1Rf11UWuVr9LaSt1xR8tXQNTigiDgBqiABCGCBIhhDVk/vz/uDQwxmQzkJjMJn2cf9zF3zpx775kpfnKWe86VmeGcc67h0pJdAOecayk8oDrnXEQ8oDrnXEQ8oDrnXEQ8oDrnXEQ8oDrnXEQ8oDrn9huSrpK0RNJiSY9KaiUpV9IsScvD104x+a+XtELSx5JG1Xv+/fU+1LzcdOvTKzPZxUhZy5d2SHYRUp5l+b+feHaWFVNWvl0NOceo09rapqLKhPIufL90hpmdWdfnknoArwODzaxE0lRgOjAYKDKz2yRNADqZ2XWSBgOPAsOBA4GXgIFmVmeBMhL9Yi1Nn16ZzJ/RK9nFSFlnH/W1ZBch5VX17pbsIqS0txbf1+BzbCqqZP6M3gnlTe++PC+BbBlAa0nlQBtgLXA9cGr4+RTgVeA64FzgMTMrBfIlrSAIrnPrOrk3+Z1zKcuAqgT/B+RJWhCzjdvjXGafAX8CVgPrgM1mNhPoZmbrwjzrgK7hIT2ANTGnKAjT6rTf1lCdc6nPMMrrbmHXtNHMhtX1Ydg3ei7QFygGnpD0/Tjnq627Im4fqQdU51xKC2ufURgJ5JvZBgBJTwMnAOsldTezdZK6A4Vh/gIgtl+wJ0EXQZ28ye+cS1mGUWmJbQlYDRwvqY0kAWcAHwLTgLFhnrHAc+H+NGCMpGxJfYEBwPx4F/AaqnMupVXFb2UnzMzmSXoSWARUAO8Ak4F2wFRJlxAE3dFh/iXhnQBLw/zj443wgwdU51wKM6AyooAKYGY3AjfWSC4lqK3Wln8iMDHR83tAdc6ltKhqqE3BA6pzLmUZUN6MJh95QHXOpSzDIm3yNzYPqM651GVQ2XziqQdU51zqCmZKNR8eUJ1zKUxU1jphKTV5QHXOpaxgUMoDqnPONVhwH6oHVOeci0SV11Cdc67hvIbqnHMRMURlM1rDyQOqcy6leZPfOeciYIgyS092MRLmAdU5l7KCG/u9ye+cc5HwQSnnnIuAmag0r6E651wkqryG6pxzDRcMSjWfMNV8Suqc2+/4oJRzzkWo0u9Ddc65hvOZUs45F6GqZjTK33xK6pzb7wSLo6QltNVH0iBJ78ZsWyRdKSlX0ixJy8PXTjHHXC9phaSPJY2q7xoeUJ1zKcsQ5Zae0Fbvucw+NrOhZjYUOAbYATwDTABmm9kAYHb4HkmDgTHAEOBMYJKkuBfygOqcS1lmUGlpCW176QzgEzP7FDgXmBKmTwHOC/fPBR4zs1IzywdWAMPjndT7UJvI05O78H+P5CJB30N2cvUdq9lZksYtl/VhfUEW3XqW8ev7VtE+pxKAlUtbcdd1vdi+NY20NPjr9GVktWpGj3/cS1feuIThp2yguCiLn44+YY/PvvWDVfzoF8sZc9pX2FKctSu9ywEl3PvUXB6+tx9PP9iniUvctPLytnPtlW/SKacEMzF9xgCee+EQTj7hU77/3ffp1XMzP7/2LJav6AxA+/al3HDdHAb238Ssl/sxaXLcOJDCtDc39udJWhDzfrKZTa4j7xjg0XC/m5mtAzCzdZK6huk9gLdijikI0+qUcgFV0h3Ap2Z2Z/h+BrDGzH4Uvv8zsBkoM7PbklfSxG1cl8mz9+fxt1c/Iru18ftLD+LV5zqxelk2R520lQuuKOTxv3bl8bu78qMb1lFZAX+44iCuvetTDh6yky1F6aRnttxgCvDS8wfy/OO9uPp3i/dIz+u2k6OOL6JwXasvHTPummUseKNzUxUxqaoqxd/+cTQrVnamdety/vrn6bzz3gGsWp3D7247hZ/9ZN4e+cvK0vnXw0dy0EHF9OldnKRSN5zB3tQ+N5rZsPoyScoCzgGury9rHUWqUyo2+d8ETgCQlAbkEfRhVDsBmNFcgmm1ygpRujONygooLUmjc7dy5s7oyMjziwAYeX4Rc1/sCMDC/7Sn76ElHDxkJwAdcitJbz4rmO2TxYs6sXVz5pfSx13zMf/4ywCsxj/jEacWsq6gNas/addEJUyuoi/asGJl8MejpCSTNQUd6ZxbwpqCjhR81vFL+UtLM1jyYVfKy5r/P5yoBqVinAUsMrP14fv1kroDhK+FYXoB0CvmuJ7A2ngnTsWA+gZhQCUIpIuBrZI6ScoGDgWOlHQ3gKQHJN0l6U1JKyV9JznFrlte93K+85NCfnDsYL479DDatq/kmFO38sXGTDp3qwCgc7cKijcFDYaCla2Q4Fff7cf4rw1k6j1d452+xTruK4VsKswmf1n7PdKzW1XynYtX8ch9/ZJUsuTq1nUbB/cr4uNlLb92bogqS2zbC99ld3MfYBowNtwfCzwXkz5GUrakvsAAYH68E6dck9/M1kqqkNSbILDOJei3GEHQ1H8fKKtxWHfgJOAQgh/hydrOLWkcMA6gd4+m++pbi9OZO6MjU+YtpV2HSn4/ri+zn+pUZ/7KClg8vy1/nb6M7NZVTLigPwOO2MFRJ29rsjInW3arSsZcks+vf3r0lz77/k8+4dmHerOzJOX++Ta6Vq3KueG6Odz392HsKMmq/4BmLniMdHT/P0tqA3wVuDQm+TZgqqRLgNXAaAAzWyJpKrAUqADGm1llvPOn6r/I6lrqCcDtBAH1BIKA+mYt+Z81sypgqaRudZ007KCeDDDsyKYb4XnntXYc0KuMnM7B/xcnnl3M0gVt6ZRXzqb1GXTuVsGm9RnkdA5qq126l3PEiO10DPMfe/oWVnzQer8KqN177qBbjxLueTwYE8jrWspdj8zjqh8MZ9Bhmzlp5Hp+eOVy2ravwKqgrCyNFx7vneRSN6709Cr+Z8IcXvlPH954q2V/190U6XqoZrYD6FwjbRPBqH9t+ScCExM9f6oG1Op+1MMJmvxrgKuBLcA/qPGDAKUx+yk38bdrj3I+XNSGnTtEdmvj3dfbM/CIHbRqXcVLU3O54IpCXpqay4hRmwE45tStPDGpKzt3iMws4/257fjWuA1J/hZNa9WK9nzvjFN3vf/nv1/j5xcex5biLH55ybG70i+89BNKdqS3+GAKxlVXzGX1mo48PW1wsgvTZIzmNVMqVQPqGwQBdGVYxS6SlEPQp/pj4BvJLNzeOuToHZz89c2MHzWI9Ayj/2ElnPX9TezcnsbEy/rw4mOd6dojuG0KoH1OJd+6dANXnD0QCYafvoXjRm5J7pdoZL+89X2OOOYLOuSU868X5/DQvQcz89m4d6jsV4YcuoGRp+WTvyqHe+74NwAPPDSUzMxKfvLjBXTsuJOb/+cVVuZ34tc3BZWtKZOfoU2bcjIyqhhxXAG/vul0Vq/JSebX2CfNacV+Wc3h0xQQzkb4ArjLzG4I0x4ARpjZIEkXAcPM7PIw/QUzezLMt83M6h36HXZkK5s/o1d92fZbZx/1tWQXIeVV9a6zd8kBby2+jy3bP2tQNOwxJMd+OvWkhPLecNi/FyZy21RjSskaalgr7VAj7aKY/QeAB2qmh+/3j/tonNsPBINSzefWr5QMqM45F/BnSjnnXCSCQanm04fqAdU5l9J8gWnnnItA9Uyp5sIDqnMupflD+pxzLgJmUF7lAdU55xosaPJ7QHXOuUg0p5lSHlCdcynLb5tyzrnIeJPfOecisxfPlEo6D6jOuZQVjPL7XH7nnGswv7HfOeci5E1+55yLgI/yO+dchHyU3znnImAmKjygOudcNJpTk7/5hH7n3H6nug81kS0RknIkPSnpI0kfShohKVfSLEnLw9dOMfmvl7RC0seSRtV3fg+ozrmUFmVABf4CvGhmhwBHAh8CE4DZZjYAmB2+R9JgYAzB05bPBCaFDxCtkwdU51zKqr4PNYqAKqkDcApwP4CZlZlZMXAuMCXMNgU4L9w/F3jMzErNLB9YAQyPdw0PqM65lFaFEtqAPEkLYrZxNU7VD9gA/FPSO5L+Lqkt0M3M1gGEr13D/D2ANTHHF4RpdfJBKedcyjKDisQXmN5oZsPifJ4BHA1cYWbzJP2FsHlfh9qqvRavAF5Ddc6ltAj7UAuAAjObF75/kiDArpfUHSB8LYzJ3yvm+J7A2ngX8IDqnEtZUfahmtnnwBpJg8KkM4ClwDRgbJg2Fngu3J8GjJGULakvMACYH+8a3uR3zqU0i/Y+1CuAhyVlASuBiwkqllMlXQKsBkYH17UlkqYSBN0KYLyZVcY7uQdU51xKi3JxFDN7F6itn/WMOvJPBCYmen4PqM65lGXWvGZKeUB1zqUwUemPkXbOuWhE3IfaqPbbgLpkfRcOv+OnyS5Gyjpww7z6M+3nMjIzk12ElKbyigafw9dDdc65qFjQj9pceEB1zqU0fwSKc85FwHxQyjnnouNNfueci4iP8jvnXATMPKA651xk/LYp55yLiPehOudcBAxR5aP8zjkXjWZUQfWA6pxLYT4o5ZxzEWpGVVQPqM65lNYiaqiS/kqcvw1m9rNGKZFzzoUMqKpqAQEVWNBkpXDOudoY0BJqqGY2Jfa9pLZmtr3xi+Scc7s1p/tQ673BS9IISUuBD8P3R0qa1Oglc845CGupCWwpIJE7Zu8ERgGbAMzsPeCUxiyUc84FhFliW0Jnk1ZJ+kDSu5IWhGm5kmZJWh6+dorJf72kFZI+ljSqvvMnNAXBzNbUSIr7bGrnnItM9DXU08xsqJlVP056AjDbzAYAs8P3SBoMjAGGAGcCkySlxztxIgF1jaQTAJOUJekawua/c841KgOrUkJbA5wLVI8ZTQHOi0l/zMxKzSwfWAEMj3eiRALqZcB4oAfwGTA0fO+cc01ACW7kSVoQs42r5WQGzJS0MObzbma2DiB87Rqm9wBiW+cFYVqd6r2x38w2AhfWl8855xpF4s35jTHN+LqcaGZrJXUFZkn6KE7e2qq9cUuTyCh/P0nPS9ogqVDSc5L61Xecc85FIsI+VDNbG74WAs8QNOHXS+oOEL4WhtkLgF4xh/cE1sY7fyJN/keAqUB34EDgCeDRxIrvnHMNUH1jfyJbPSS1ldS+eh/4GrAYmAaMDbONBZ4L96cBYyRlS+oLDADmx7tGInP5ZWYPxrx/SNLlCRznnHMNFuGN/d2AZyRBEPseMbMXJb0NTJV0CbAaGB1c15ZImgosBSqA8WYW9w6neHP5c8PdVyRNAB4j+HtxAfDvBn0t55xLVERz+c1sJXBkLembgDPqOGYiMDHRa8SroS4kCKDV3+bS2OsAv0v0Is45t6+UIrOgEhFvLn/fpiyIc859SQpNK01EQuuhSjoMGAy0qk4zs381VqGccy6Q2IBTqqg3oEq6ETiVIKBOB84CXgc8oDrnGl8zqqEmctvUdwg6bD83s4sJOnWzG7VUzjlXrSrBLQUk0uQvMbMqSRWSOhDc9Oo39u+FrPQKHjj/ObLSK0lPq2LW8n5MmjucP549kz6digFon13G1tIsRj98PgAD8zbxmzP+Q9vsMszEmEe+TVlly31izS/+9CnHjdxM8cYMLh05GID2ORX8alI+3XqVsX5NFhN/0pdtmzPo1rOUv726lIJPgh6ojxa15a7reyez+E3i5ze8z/CTCin+Iovx3w0WfPvej5cx6tw1bCnOAmDKpEEseDOYOdmn/xYuv34xbdpWYFVw5UUnUl4Wd22P1NNSFpiOsUBSDvA3gpH/bdRzc2siJFUCHwCZBPd4TQHuNLMU+VsTnbLKdC558hxKyjPJSKtkyvnP8np+b66d/rVdea455U22lQb/UaSrilvPfInrXzyDZRvz6NhqJxXN6Nnk+2LmE7lMe6AL1965alfa+eM/55032jP1ngM4f/znXDB+PfffEkylXrcqm5+OOjRJpU2Ol/7dkxeeOIhf3PTeHunPPdqXpx/es46Tll7FNb99jz/fdCT5yzvQvmMZlRXN899Qcxrlr/cXNrOfmlmxmd0LfBUYGzb9G6okXEJrSHjes4Eba2aS1AKqZaKkPBOAjLQqMtKqsD2mCRujBq5g+sf9ATjhoDUs29iZZRvzANi8sxVV1jz/Y0jU4nnt2Vq8Z+1pxNc289ITnQF46YnOjBhVnIyipYwl7+SydUtmQnmPPm4jq1a0J395BwC2bs5qVs9m2kMzWmA63o39R8f7zMwWRVUIMysMV355W9JNBNO/vk5wV0FbSecQTAfrRFCjvcHMnpP0S2Cnmd0l6Q7gSDM7XdIZwMVm9v2oythQaari8e89Se+czTz23mF88Hm3XZ8d02Mdm3a0YXVxDgAHdSrGEPd+8wU6tS7hxWX9+eeCo5JV9KTplFdBUWEQQIoKM8npXLHrswN6l3HPix+yY1s6U/5wIIvnt0tWMZPuG6M/5fSzP2P5hx25/y+Hsm1rJj16b8cMbr5rPh1zypgzqztPPXhwsova4sWr/f05zmcGnB5lQcxspaQ0di+dNQI4wsyKwlrqN81si6Q84C1J04A5wNXAXcAwIFtSJnAS8FrNa4RBexxAZodONT9uVFWWxuiHz6d9dil3/teL9O+8iRWbgtrXWYOWM/2j/rvypqcZRx24ju8+8m12VmTw928/z9L1XZi3pmeTljlVFRVm8v3hh7G1OIP+h+/gpvs/Ydzpg9mxrZn1D0Zg+lMH8dj9AzCDH1y2jEt+/iF/+f0RpKcbg4d+wVVjT6R0ZzoTJ81jxUcdee/tvGQXea+1iCa/mZ0WZ4s0mMaIbZPMMrOimPRbJL0PvESwJmE3gj7dY8IFD0qBuQSB9WRqCahmNtnMhpnZsPTWbRvpK8S3tTSbtwsO5MQ+wTKL6apiZP98ZizbHVDXb23LwoIDKd7Zmp0Vmby2qjeHdt2QlPIm0xcbM8jtWg5AbtdyijcFf//Ly9LYWhzsr/igDWs/zaZHv51JK2cyFRdlU1UVPALkxWd7MXBI0C2ysbAVixflsmVzFqWl6Sx4owsHD9qS5NLuAyOYeprIlgJSpmMuXBKwkt1LZ8U+YfVCoAtwjJkNBdYDrcysHFgFXAy8SRBETwMOJoWeKtCpdQnts0sByE6v4PjeBeQXBc3743sXkP9FDuu37W6yvvlpbwbkbaJVRjnpqmJYz7V8UpRb67lbsrdmdWTk6E0AjBy9ibkzOwLQMbectLSg2nJA71J69C3l89X75518nTrv/kNywqnr+fST9gAseqsLffpvJTu7krT0Kg4/uog1+c20W6Ql9KE2JUldgHuBu83MwtVgYnUECs2sXNJpwEExn80BrgF+SHDXwO3AQrPUefhsl7Y7+P2ol0lXFZIxc1l/5uT3AeCsQSuY/vGAPfJvKc3mwUVH8uj3nsIMXlt1EK/lH1TLmVuOCXfnc8SIrXTMreChtz/gwT935/G7D+DX9+Zz5phNFH6WxcTLgtnQhx+/jf++eh2VlaKyEu6a0GtXjbUl++Xv3uHwY4rokFPGlOdf5uG/DeDwozfRb+AWzEThutb89dbDANi2NZNnH+nLHVPewAwWvNmVt9/oWs8VUlNzavIrWXGnltumHgRuD+95vQgYZmaXh3nzgOfDvO8CJwJnmdmqcADqRSDHzLZLWgbca2a3x7t+6wN6Wb+xv2ikb9f8HfjneckuQsrLOPCAZBchpb35+aNsLlvfoLZ4dq9e1vPKqxLKu/KaqxcmsGJ/o0pk6qkImtz9zOxmSb2BA8ysQfeimlmdIwhm9gDwQMz7jQSDVLXlnU0QaKvfD2xIuZxzKaYZ1VAT6UOdRBDMvhu+3wrc02glcs65kCzxLRUk0vF0nJkdLekdADP7QlJWI5fLOecCKTKCn4hEAmq5pHTCinc4gNTipoc651JTqtQ+E5FIk/8ugqcDdpU0kWDpvlsatVTOOVetJd02ZWYPS1pIsISfgPPMLGXu8XTOtWAp1D+aiERG+XsDOwhuW9qVZmarG7NgzjkHpEztMxGJ9KH+m90P62sF9AU+BoY0Yrmccw4ANaMRm0SW7zvczI4IXwcAwwn6UZ1zrlmRlC7pHUkvhO9zJc2StDx87RST93pJKyR9LGlUIuff67n84bJ9x+7tcc45t0+iHZT6OXuu8zEBmB1WFmeH75E0GBhD0BI/E5gU3u0UVyJ9qLHzM9OAo4H9b+kj51zTi3BQSlJPgnWWJwLVce1cgoeQQvDUkFeB68L0x8ysFMiXtIKgdT433jUS6UNtH7NfQdCn+lRC38A55xoq8YCaJ2lBzPvJZjY55v2dwC/ZM6Z1M7N1AGa2TlL1CjI9gLdi8hWEaXHFDahhFbedmV1b34mcc65RJB5QN9a1OIqkbxCsWLdQ0qkJnKu26Vn1liTeI1AyzKwi3qNQnHOuMYnIRvlPBM6RdDbB3UodJD0ErJfUPayddmf3eswFQK+Y43sCa+u7SLxBqerVpN6VNE3SDyR9q3rb66/jnHN7K6LFUczsejPraWZ9CAabXg6fOTeN4Bl2hK/PhfvTgDGSsiX1BQaQwNOeE+lDzQU2ETxDqvp+VAOeTuBY55xrmMa9sf82YKqkS4DVwGgAM1siaSqwlGDsaLyZVdZ3sngBtWs4wr+Y3YG0WjOau+Cca9YijjZm9irBaD5mtolgWn1t+SYS3BGQsHgBNR1oxz52zjrnXBRaylz+dWZ2c5OVxDnnatNCAmrzWdXVOdcyWfOayx8voNbar+Ccc02qJdRQzayoKQvinHO1aSl9qM45l3weUJ1zLgIp9HiTRHhAdc6lLOFNfueci4wHVOeci4oHVOeci4gHVOeci0BLe4y0c84llQdU55yLRkuZetqiqRKyi5vRn76mVlXv0o/7vS3Deya7CCmtcnZWJOfxJr9zzkXBb+x3zrkIeUB1zrmG85lSzjkXIVU1n4jqAdU5l7q8D9U556LjTX7nnItKMwqoackugHPOxSNLbKv3PFIrSfMlvSdpiaTfhum5kmZJWh6+doo55npJKyR9LGlUfdfwgOqcS22W4Fa/UuB0MzsSGAqcKel4YAIw28wGALPD90gaDIwBhgBnApMkpce7gAdU51zqCp96mshW76kC28K3meFmwLnAlDB9CnBeuH8u8JiZlZpZPrACGB7vGh5QnXMpq/o+1ASb/HmSFsRs4750Pild0rtAITDLzOYB3cxsHUD42jXM3gNYE3N4QZhWJx+Ucs6lNkt4VGqjmQ2LfyqrBIZKygGekXRYnOyq7RTxzu81VOdcSotqUCqWmRUDrxL0ja6X1B0gfC0MsxUAvWIO6wmsjXdeD6jOudSV6IBUYqP8XcKaKZJaAyOBj4BpwNgw21jguXB/GjBGUrakvsAAYH68a3iT3zmX0iJcD7U7MCUcqU8DpprZC5LmAlMlXQKsBkYDmNkSSVOBpUAFMD7sMqiTB1TnXEqLKqCa2fvAUbWkbwLOqOOYicDERK/hAdU5l7qMvRmUSjoPqM65lOZz+Z1zLioeUJ1zruF8gWnnnIuKmS8w7ZxzkWk+8dQDqnMutXmT3znnomCAN/mdcy4izSeeekB1zqU2b/I751xEfJTfOeei4I+Rds65aAQ39jefiOoB1TmX2qJbvq/ReUB1zqU0r6E651wUvA/V1ZSVUcHfLn6OzIwq0tOqmL20H5NfOZbLTp/PVwatosrEF9tbc9Ozp7Fxa1s6tt7J/7tgJoMPLOSFdwfxh+knJ/srNLnzLtnAWRcWIRn/93Bnnvl7F/oNKeFntxWQ1aqKygpx9/U9+fjdNskuapPomrONX//3K+R2KMFMTHvjEJ589XBuuvglenfbDEC71qVsK8nmh7d9m/S0Kq678D8M7LWR9DRjxvwBPDTzS2srNwM+lx8ASZXABwTPvq4geN71nWYWt0dE0h+Bs4HpZnbtXl5zKHCgmU3ft1I3jrKKdC6bcg4lZZmkp1Vy/yXP8eby3jz4xlDufTl4zPcFx33Aj7+ykFtfOIXSinT+9+Vj6d+1iIO7FiW59E3voEElnHVhET/7+gDKy8Qtj6xk3uwO/OiGtTx0ezcWvNKBY0/fwiU3rOWX3+mf7OI2icqqNO55egTLCvJonV3G/dc9w4KPenLTP0fuyjP+m3PZXpIFwGlHryQro5KLbhlNdmYFD94wlZcW9OfzovbJ+gr7rhk1+RvzIX0lZjbUzIYAXyUIkjcmcNylwNF7G0xDQ8PrpBhRUpYJQEZ6FRlpVZjB9tKsXTlaZ5XvatnsLM/kvdXdKa1IT0JZk6/3gFI+XNSG0pI0qirF+3PbceJZmzGDtu2DR/q07VBJ0frMJJe06Wza0oZlBXkAlJRmserzHPJytsfkME47eiUvLQz+wJhBq6wK0tOqyM6qoKIyne07m+HvZcEjUBLZUkGTNPnNrFDSOOBtSTcRBPLbgFOBbOAeM7tP0jSgLTBP0q3Ay8C9QO/wVFea2RuShgN3Aq2BEuBiIB+4GWgt6STgVjN7vCm+XyLSVMWDlz5Fr9zNPPH2YSz5rBsAPz1jHmcfuYztO7O49IFzklzK1LDqo1ZcdN062neqoGxnGseevoXl77fm3t/04JZHV/Lj36xDMq46Z0Cyi5oUB+RuZWDPjSxd1XVX2pEHf84XW1tTsKEjAK++04+Tj1jFsxMfIjurgr8+PYKtO1olq8gN04xqqE3Wh2pmKyWlAV2Bc4HNZnaspGzgDUkzzewcSdvMbCiApEeAO8zsdUm9gRnAoQSPfj3FzCokjQRuMbNvS/oNMMzMLq+tDGFQHweQ2a5TY3/lPVRZGhfeO5p2rUr505gZHNy1iE8Kc5k0+zgmzT6Oi05exPnHLWbyK8c2ablS0ZoVrZg6qSu3PraSndvTyF/amsoK8Y2xm7jvxgN5fXoOp/xXMb+4fQ0TLjg42cVtUq2zyvn9j2Zx11MnsGPn7hbOyGEreGnB7u6PwX0KqaxK47xff5/2bUq556ppLPioB+s2dUhGsRum+cTTRm3y10bh69eA/5b0LjAP6EzwzOuaRgJ3h/mmAR0ktQc6Ak9IWgzcAeh4GEoAAAyBSURBVAxJ5OJmNtnMhpnZsIxWbRv4VfbNtp3ZLFx1ICP6r94j/cX3B3DGoSuTUqZUNOPRzlw+aiDXfKs/W4vT+Sw/m6+OLuL16UENbM7zHRk4dEeSS9m00tOq+P2PZzFrQX/mvNd3j/RTjlzFy4v67UobOWwF85f2pLIqjeJtrflgZTcO6b0hGcVuMFVVJbTVex6pl6RXJH0oaYmkn4fpuZJmSVoevnaKOeZ6SSskfSxpVH3XaLKAKqkfUAkUEgTWK8I+1qFm1tfMZtZRvhEx+XqY2Vbgd8ArZnYY8F9ASrdlctqU0K5VKQDZGRUM71fAqo2d6JVbvCvPVw5ZxaqNTVtrTmUdO5cD0KVHGSeevZlXn81h0/pMjhgR9BsOPWkba/Ozk1nEJmZMuPA/rPo8h8dfPmKPT44Z9Bmr1+ewobjdrrT1Re04etBawGiVVc6QPoWsXp/TxGWOgBHc2J/IVr8K4GozOxQ4HhgvaTAwAZhtZgOA2eF7ws/GEFTYzgQmSYo7sNEkTX5JXQj6Qu82M5M0A/iJpJfNrFzSQOAzM9te49CZwOXAH8PzDDWzdwlqqJ+FeS6Kyb8VSLlhzLz2O/jtN18mTUaajFlLDub1ZQfxhwtmcFDnYqpMrNvcnluf33171LQrH6JtdjmZ6ZV85ZBVXP7g18nfkJvEb9G0fvP3T2nfqYLKcnH3r3qwbXMGd17bk5/cvJb0dKOsNI07r+2Z7GI2mcP7refM45bzyWe5/GPCUwBMnnYsby3tzchjPuGlhXt2fTwzZwjXf/9V/vXrJxHG9LcG8cnazskoeoMIi+zGfjNbB6wL97dK+hDoQdAFeWqYbQrwKnBdmP6YmZUC+ZJWAMOBuXWW1xqpw7eW26YeBG43s6qwL/X3BLVLARuA88xsc9iH2i48Rx5wD0G/aQYwx8wukzSC4ItvIBi4+oGZ9ZGUS9DPmkk9g1JtuvSyQ751VaN895ag89/q/DfjQju+dVyyi5DS3pv9F7Z9sUb156xbx7YH2vGHjkso78yFv/0U2BiTNNnMJteWV1IfYA5wGLDazHJiPvvCzDpJuht4y8weCtPvB/7PzJ6sqwyNVkM1szqrxuG9qL8Kt5qftYvZ3whcUEueucDAmKT/CdOLAB/Vca4lSbzSt9HMhtWXSVI74CmCu4a2SHXG/No+iFuYph6Ucs65xEXbh4qkTIJg+rCZPR0mr5fUPfy8O8E4D0AB0Cvm8J7A2njn94DqnEtpEY7yC7gf+NDMbo/5aBowNtwfCzwXkz5GUrakvgR3Is2Pdw2fy++cS2EW5Y39JwI/AD4Ib8WEoNvxNmCqpEuA1cBoADNbImkqsJRgHGi8mVXGu4AHVOdc6jIiC6hm9jq194sCnFHHMROBiYlewwOqcy61pcg8/UR4QHXOpTRfYNo556LiAdU55yJgBpXNp83vAdU5l9q8huqccxHxgOqccxEwwJ8p5ZxzUTCI/xi6lOIB1TmXugwflHLOuch4H6pzzkXEA6pzzkUh0sVRGp0HVOdc6jIggaX5UoUHVOdcavMaqnPORcGnnjrnXDQMzO9Ddc65iPhMKeeci4j3oTrnXATMfJTfOeci4zVU55yLgmGVcR80mlI8oDrnUlczW74vLdkFcM65uKwqsa0ekv4hqVDS4pi0XEmzJC0PXzvFfHa9pBWSPpY0KpGiekB1zqUsA6zKEtoS8ABwZo20CcBsMxsAzA7fI2kwMAYYEh4zSVJ6fRfwgOqcS11mkdVQzWwOUFQj+VxgSrg/BTgvJv0xMys1s3xgBTC8vmt4H6pzLqXtxaBUnqQFMe8nm9nkeo7pZmbrAMxsnaSuYXoP4K2YfAVhWlz7bUAt2Viw8Z3JV3+a7HLEyAM2JrsQKS61fqOnnkx2CWpKrd8HDmroCbbyxYyX7Mm8BLNvNLOaTfp9pVrS6u1X2G8Dqpl1SXYZYklaYGbDkl2OVOa/UXwt8feJMEDWZb2k7mHttDtQGKYXAL1i8vUE1tZ3Mu9Ddc7tz6YBY8P9scBzMeljJGVL6gsMAObXd7L9tobqnNu/SHoUOJWgr7UAuBG4DZgq6RJgNTAawMyWSJoKLAUqgPFmVm9nrqwZTetqySSNS6ADfb/mv1F8/vsknwdU55yLiPehOudcRDygOudcRDygNgFJd0i6Mub9DEl/j3n/Z0m/kTQhOSVsepIqJb0raYmk9yT9QtJ+/e9xX38TSX8Mj/njPlxzqKSz963EriYf5W8abxKMHt4Z/geSB3SI+fwE4Eozm5eMwiVJiZkNBQhnpzwCdCQYed1FUoaZVSShfMmQ0G9Si0uBLmZWug/XHAoMA6bvw7Guhv26RtCE3iAImhAstrAY2Cqpk6Rs4FDgSEl3A0h6QNJdkt6UtFLSd5JT7KZhZoXAOOByBS6S9ISk54GZktpJmi1pkaQPJJ0LIOmXkn4W7t8h6eVw/wxJDyXtC0Wglt8kPayJvi3pfUmXAkiaBrQF5km6QFIXSU+F+d6WdGKYb3j47+md8HWQpCzgZuCCsGZ8QbK+b0vhNdQmYGZrJVVI6k0QWOcSzAseAWwG3gfKahzWHTgJOITgJuOUm+cYJTNbGdbeq+dSjwCOMLMiSRnAN81si6Q84K0wkMwBrgbuIqhlZUvKJPjdXmv6bxGtGr/JucBmMzs2/CP8hqSZZnaOpG0xNdtHgDvM7PXw39sMgj/YHwGnmFmFpJHALWb2bUm/AYaZ2eVJ+ZItjAfUplNdSz0BuJ0goJ5AEFDfrCX/sxY8P3eppG5NVsrkip0/PcvMimLSb5F0ClBF8Nt1AxYCx0hqD5QCiwgC68nAz5qs1I2r+jf5GnBETGulI8Hsnfwa+UcCg6VdP2WH8PfpCEyRNIBgTnpmo5Z6P+UBtem8SRBADydo8q8hqF1tAf4BdK6RP7Y/rLaFGloUSf2ASnbPpd4e8/GFQBfgGDMrl7QKaBWzfzHB7/s+cBpwMPBhExW90dT4TQRcYWYz6jksDRhhZiU1zvVX4BUz+6akPsCrkRfYeR9qE3oD+AZQZGaVYe0rh6BpOzepJUsySV2Ae4G7rfaZJh2BwjCAnsaeqxjNAa4JX18DLgPereM8zUYtv8kM4CdhlwaSBkpqW8uhM4HLY84zNNztCHwW7l8Uk38r0D7a0u+/PKA2nQ8IRvffqpG22cxSacm1ptK6+hYh4CWCQPDbOvI+DAxTsNblhQT9gdVeI+hvnmtm64GdNN/+03i/yd8J5pUvUvAIj/uovYX5M4Lf6n1JSwn+wAD8AbhV0htA7MrzrxB0EfigVAR86qlzzkXEa6jOORcRD6jOORcRD6jOORcRD6jOORcRD6jOORcRD6iuVjErHy0O59W3acC5Hqie4SPp75IGx8l7qqQT6vo8znGrwmmpCaXXyLNtL691k6Rr9raMruXzgOrqUmJmQ83sMIJ1Bi6L/VBSeu2HxWdmPzKzpXGynMruhWSca1Y8oLpEvAb0D2uPr4QLcHwQZwUkSbpb0lJJ/2b3gidIelXSsHD/zHAFqffC1aT6EATuq8La8clxVk/qLGlmuHrSfSQwPVfSs5IWKlg7dFyNz/4clmV2OEsJSQdLejE85jVJh0TxY7qWy+fyu7jClZ7OAl4Mk4YDh5lZfhiUvrQCEnAUMIhg3YJuBDN8/lHjvF2AvxGsgJQvKTdcWepeYJuZ/SnMV9fqSTcCr5vZzZK+TrDUXX1+GF6jNfC2pKfMbBPB8neLzOzqcPWlGwmmb04GLjOz5ZKOAyYBp+/Dz+j2Ex5QXV1aS3o33H8NuJ+gKT7fzKpXOKprBaRTgEfDx+6uVbhOaQ3HA3OqzxWzslRNda2edArwrfDYf0v6IoHv9DNJ3wz3e4Vl3USwgtXjYfpDwNOS2oXf94mYa2cncA23H/OA6uqya/X4amFgiV0FqtYVkBQ8UqO+Oc1KIA/UvXoSCR5fnf9UguA8wsx2SHoVaFVHdguvW1zzN3AuHu9DdQ1R1wpIc4AxYR9rd4Il9WqaC3xFUt/w2NwwvebqR3WtnjSHYKEUJJ0FdKqnrB2BL8JgeghBDblaGlBdy/4eQVfCFiBf0ujwGpJ0ZD3XcPs5D6iuIepaAekZYDnBalr/C/yn5oFmtoGg3/NpSe+xu8n9PPDN6kEp6l496bfAKZIWEXQ9rK6nrC8CGZLeB37Hnqt+bQeGSFpI0Ed6c5h+IXBJWL4lBKvmO1cnX23KOeci4jVU55yLiAdU55yLiAdU55yLiAdU55yLiAdU55yLiAdU55yLiAdU55yLyP8H6GeR3ABXWtoAAAAASUVORK5CYII=\n",
   "text/plain": "<Figure size 432x288 with 2 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "              precision    recall  f1-score   support\n\n         Win       0.43      0.39      0.41       707\n        Draw       0.30      0.16      0.21       637\n      Defeat       0.53      0.69      0.60      1161\n\n    accuracy                           0.47      2505\n   macro avg       0.42      0.42      0.41      2505\nweighted avg       0.45      0.47      0.45      2505\n\n\n\nAccuracy:  0.4746506986027944\nRecall:  0.4174250091051073\nPrecision:  0.42210595641660403\nF1 Score:  0.4091664738968846\n"
 }
]
```

### Neural Network

```{.python .input  n=179}
visible = Input(shape=(features.shape[1],))
hidden = Dense(500, activation='relu')(visible)
output = Dense(3, activation='softmax')(hidden)

clf = Model(inputs=visible, outputs=output)
print(clf.summary())

from keras import metrics
from keras import losses
from keras import optimizers

clf.compile(optimizer=optimizers.Adam(), 
              loss=losses.CategoricalCrossentropy(), 
              metrics=[metrics.Precision(), metrics.Recall()])

train_predict_nn(clf, features, outcomes)
```

```{.json .output n=179}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Model: \"model_16\"\n_________________________________________________________________\nLayer (type)                 Output Shape              Param #   \n=================================================================\ninput_16 (InputLayer)        (None, 60)                0         \n_________________________________________________________________\ndense_70 (Dense)             (None, 500)               30500     \n_________________________________________________________________\ndense_71 (Dense)             (None, 3)                 1503      \n=================================================================\nTotal params: 32,003\nTrainable params: 32,003\nNon-trainable params: 0\n_________________________________________________________________\nNone\n"
 },
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 2 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "              precision    recall  f1-score   support\n\n         Win       0.46      0.36      0.41       707\n        Draw       0.00      0.00      0.00       637\n      Defeat       0.51      0.86      0.64      1161\n\n    accuracy                           0.50      2505\n   macro avg       0.32      0.41      0.35      2505\nweighted avg       0.37      0.50      0.41      2505\n\n\n\nAccuracy:  0.5001996007984032\nRecall:  0.40712963876675595\nPrecision:  0.32404323902410287\nF1 Score:  0.3490835838661926\n"
 }
]
```

### Deep Neural Network

```{.python .input  n=177}
visible = Input(shape=(features.shape[1],))
hidden1 = Dense(500, activation='relu')(visible)
hidden2 = Dense(100, activation='relu')(hidden1)
hidden3 = Dense(50, activation='relu')(hidden2)
hidden4 = Dense(20, activation='relu')(hidden3)
output = Dense(3, activation='softmax')(hidden4)

clf = Model(inputs=visible, outputs=output)
print(clf.summary())

from keras import metrics
from keras import losses
from keras import optimizers

clf.compile(optimizer=optimizers.Adam(), 
              loss=losses.CategoricalCrossentropy(), 
              metrics=[metrics.Precision(), metrics.Recall()])

train_predict_nn(clf, features, outcomes)
```

```{.json .output n=177}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Model: \"model_15\"\n_________________________________________________________________\nLayer (type)                 Output Shape              Param #   \n=================================================================\ninput_15 (InputLayer)        (None, 60)                0         \n_________________________________________________________________\ndense_65 (Dense)             (None, 500)               30500     \n_________________________________________________________________\ndense_66 (Dense)             (None, 100)               50100     \n_________________________________________________________________\ndense_67 (Dense)             (None, 50)                5050      \n_________________________________________________________________\ndense_68 (Dense)             (None, 20)                1020      \n_________________________________________________________________\ndense_69 (Dense)             (None, 3)                 63        \n=================================================================\nTotal params: 86,733\nTrainable params: 86,733\nNon-trainable params: 0\n_________________________________________________________________\nNone\n"
 },
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 2 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "              precision    recall  f1-score   support\n\n         Win       0.50      0.40      0.45       707\n        Draw       0.00      0.00      0.00       637\n      Defeat       0.53      0.88      0.66      1161\n\n    accuracy                           0.52      2505\n   macro avg       0.34      0.43      0.37      2505\nweighted avg       0.39      0.52      0.43      2505\n\n\n\nAccuracy:  0.5205588822355289\nRecall:  0.42656572781678315\nPrecision:  0.3432206964342041\nF1 Score:  0.36800706434958713\n"
 }
]
```

## Conclusion
[go back to the top](#Table-of-contents)

### What did we learn
Regarding data analysis we learn so many things we did not knew before.
Regarding data analysys and processment of data, we learned the importance of
usability of the dataset and in case it is not perfect we bust fixed it.
After looking at our dataset we noticed some inconsistences and fixes by either
removing some rows and **mean imputation** to allow the models to have multiple
features.
Before fitting the models, we noticed also, the importance of splitting in a
good manner the data or train and test and used **K-Fold Cross validation**.
Finally, the analysis of the models is very important... We cannot look only for
the accuracy measure because many times it's misleading and to counter that, we
must focus on other metrics to evaluate our models.


### Choosing best model

To evaluate the different models we need to choose a metric for comparison. The
**accuracy** is only useful when we are dealing with a balanced dataset, which
is not the case. Because of that, we need to consider only the other 3 possible
metric, **recall**, **precison** or **f-measure**.

Since a predicton is associated with a cost in our case (given that to predict a
match we have to spend money to bet on it), the most valuable measure is the
precision. Then, our objective is to try to get the maximum percentage of **true
positives** that are correctly classified by our models.

In other cases, such as for medical applications, the objective of a model is to
maximize the recall. The f1-score is also an important measure to evaluate both
the recall and the precision, when the 2 have the same importance.

Considering this, we think that both Naives Bayes and KNN were the most
effective. Both of them got the most precision alongside with Gradient Boosting
and SVC. The test were done thoroughly and exhaustive, with shuffling the data
and measuring more than once obviously. This way, by running once **the results
may vary** but not considerably.

### All in all
All in all, it was a great experience and very valuable. We noticed the interest
of the professors to shift the lecture to do a more pratical work. In that way
we can learn much more by ourselves, learn with our mistakes, become very
interested in this topic and pursue it in future years.

## References

[go back to the top](#Table-of-contents)

* [Europen Soccer Database](https://www.kaggle.com/hugomathien/soccer)

* [Football Data Analysis](https://www.kaggle.com/pavanraj159/european-football-
data-analysis#The-ultimate-Soccer-database-for-data-analysis-and-machine-
learning)

* [Data Analysis and Machine Learning Projects](https://github.com/rhiever/Data-
Analysis-and-Machine-Learning-Projects/blob/master/example-data-science-
notebook/Example%20Machine%20Learning%20Notebook.ipynb)

* [Match Outcome Prediction in Football](https://www.kaggle.com/airback/match-
outcome-prediction-in-football)

* [European Soccer Database Supplementary (XML Events to
CSV)](https://www.kaggle.com/jiezi2004/soccer)

* [A deep learning framework for football match
prediction](https://link.springer.com/article/10.1007/s42452-019-1821-5)

* [Predicting Football Match Outcome using Machine Learning: Football Match
prediction using machine learning algorithms in jupyter
notebook](https://github.com/prathameshtari/Predicting-Football-Match-Outcome-
using-Machine-Learning)

* [(PDF) Football Result Prediction by Deep Learning
Algorithms](https://www.researchgate.net/publication/334415630_Football_Result_Prediction_by_Deep_Learning_Algorithms)

* [Predicting Football Results Using Machine Learning
Techniques](https://www.imperial.ac.uk/media/imperial-college/faculty-of-
engineering/computing/public/1718-ug-projects/Corentin-Herbinet-Using-Machine-
Learning-techniques-to-predict-the-outcome-of-profressional-football-
matches.pdf)

* [A machine learning framework for sport result
prediction](https://www.sciencedirect.com/science/article/pii/S2210832717301485)

```{.python .input}

```

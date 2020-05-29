# Supervised learning models to predicting football matches outcomes

**Results** - You can see the results on the html file in src/ \
**Disclaimer** - This repository was created for educational purposes and we do not take any responsibility for anything related to its content. You are free to use any code or algorithm you find, but do so at your own risk.

### Notebook by [Martim Pinto da Silva](https://github.com/motapinto), [LuisRamos](https://github.com/luispramos), [Francisco Gonçalves](https://github.com/kiko-g)
#### Supported by [Luis Paulo Reis](https://web.fe.up.pt/~lpreis/)
#### [Faculdade de Engenharia da Universidade do Porto](https://sigarra.up.pt/feup/en/web_page.inicial)

#### It is recommended to [view this notebook in nbviewer](https://nbviewer.ipython.org/github.com/motapinto/football-classification-predications/blob/master/src/Supervised%20Learning%20Models.ipynb) for the best overall experience
#### You can also execute the code on this notebook using [Jupyter Notebook](https://jupyter.org/) or [Binder](https://mybinder.org/)(no local installation required)

## Table of contents
1. * [Introduction](#Introduction)
2. * [Required libraries and models](#Required-libraries-and-models)
    1. - [Libraries](#Libraries)
    2. - [Models](#Models)
3. * [The problem domain](#The-problem-domain)
4. * [Step 1: Data analysis](#Step-1:-Data-analysis)
    1. - [Extracting data from the database](#Extracting-data-from-the-database)
    2. - [Matches](#Matches)
    3. - [Team Stats - Team Attributes](#Team-Stats---Team-Attributes)
    4. - [Team Stats - Shots](#Team-Stats---Shots)
    5. - [Team Stats - Possession](#Team-Stats---Possession)
    6. - [Team Stats - Crosses](#Team-Stats---Crosses)
    7. - [FIFA data](#FIFA-data)
    8. - [Joining all features](#Joining-all-features)
5. * [Step 2: Classification & Results Interpretation](#Step-2:-Classification-&-Results-Interpretation)
    1. - [Training and Evaluating Models](#Training-and-Evaluating-Models)
    2. - [The basis](#The-basis)
    3. - [KNN](#KNN)
    4. - [Decision Tree](#Decision-Tree)
    5. - [SVC](#SVC)
    6. - [Naive Bayes](#Naive-Bayes)
    7. - [Gradient Boosting](#Gradient-Boosting)
    8. - [Neural Network](#Neural-Network)
    9. - [Deep Neural Network](#Deep-Neural-Network)
6. * [Conclusion](#Conclusion)
    1. - [What did we learn](#What-did-we-learn)
    2. - [Choosing best model](#Choosing-best-model)
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

```python
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

#### Did you specify the type of data analytic question (e.g. exploration, association causality) before touching the data?

> We are trying to design a predictive model capable of accurately predicting if
> the home team will either win, lose or draw, i.e., predict the outcome of
> football matche based on a set of measurements, including player ratings, team
> ratings, team average stats(possession, corners, shoots), team style(pressing,
> possession, defending, counter attacking, speed of play, ..) and team match
> history(previous games).

#### Did you define the metric for success before beginning?

> Let's do that now. Since we're performing classification, we can use
> [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision) - the fraction
> of correctly classified matches - to quantify how well our model is performing.
> Knowing that most bookkeepers predict matches with an accuracy of 50%, we will
> try to match or beat that value. We will also use a confusion matrix, and
> analyse the precision, recall and f1-score.

#### Did you consider whether the question could be answered with the available data?"

> The data provided has information about more than 25k matches across multiple
> leagues. Even though the usability isn't great, after some processing and
> cleansing of the data, we will be able to predict matches with great confidence.
> To answer the question, yes, we have more than enough data to analyse football
> matches..

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

```python
with sqlite3.connect("../dataset/database.sqlite") as con:
    matches = pd.read_sql_query("SELECT * from Match", con)
    team_attributes = pd.read_sql_query("SELECT distinct * from Team_Attributes",con)
    player = pd.read_sql_query("SELECT * from Player",con)
    player_attributes = pd.read_sql_query("SELECT * from Player_Attributes",con)
```

### Matches
We start by cleaning the match data and defining some methods for the data
extraction and the labels

```python
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

```python
viable_matches = matches
viable_matches.describe()
```

Looking at the match data we can see that most columns have 25979 values. This
means we are analysing this number of matches from the database. We can start by
looking at the bookkeeper data. We can see that the number of bookkepper match
data is different for each bookkeper. We start by selecting the bookeeper with
the most predictions data available.

```python
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

Analysing the description of the dataframe, we can see that the bookkeper
regarding Bet 365 has the most available information and, has such, we will
decide to selected it as a feature input for our models.

We also need to consider that some of these matches may not be on the team
attributes that we will clean after this. In that case, we need to remove any
matches that does not contain any team stats information, since **mean
imputation** would't work in these case.

We also need to remove some rows that do not contain any information about the
position of the players for some matches.

```python
teams_stats = team_attributes
viable_matches = viable_matches.dropna(inplace=False)

home_teams = viable_matches['home_team_api_id'].isin(teams_stats['team_api_id'].tolist())
away_teams = viable_matches['away_team_api_id'].isin(teams_stats['team_api_id'].tolist())
viable_matches = viable_matches[home_teams & away_teams]

viable_matches.describe()
```

### Team Stats - Team Attributes

```python
teams_stats.describe()
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

```python
teams_stats['buildUpPlayDribbling'].hist();
```

We can see that most buildUpPlayDribbling values fall within the 45 - 55 range,
so let's fill in these entries with the average measured buildUpPlaySpeed

```python
build_up_play_drib_avg = teams_stats['buildUpPlayDribbling'].mean()
# mean imputation
teams_stats.loc[(teams_stats['buildUpPlayDribbling'].isnull()), 'buildUpPlayDribbling'] = build_up_play_drib_avg
# showing new values
teams_stats.loc[teams_stats['buildUpPlayDribbling'] == build_up_play_drib_avg].head()
```

```python
teams_stats.loc[(teams_stats['buildUpPlayDribbling'].isnull())]
```

Having done the **mean imputation** for team_attributes we can see that there
are no longer missing values for the buildUpPlayDribbling. After that, we
decided to select only continuous data, i.e, select only columns that "store"
numerical values that we will provide to the input of the supervised learning
models.

```python
teams_stats.drop(['buildUpPlaySpeedClass', 'buildUpPlayDribblingClass', 'buildUpPlayPassingClass', 
    'buildUpPlayPositioningClass', 'chanceCreationPassingClass', 'chanceCreationCrossingClass',  
    'chanceCreationShootingClass','chanceCreationPositioningClass','defencePressureClass', 'defenceAggressionClass', 
    'defenceTeamWidthClass','defenceDefenderLineClass'], inplace = True, axis = 1)

teams_stats.describe()
```

### Team Stats - Shots
After cleaning the team attributes data we need to consider adding some more
stats to each match. We will start by adding the average of the number of shots
per team. The number of shots consists on the sum of the shots on target and the
shots of target. After merging all the information to teams_stats we have to
analyse the data again.

```python
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

As we can see, there are a lot of Nan values on the avg_shots column. This
represents teams that did not have shots data on this dataset. Instead of
removing thoose rows, and give less input to our models we need again to do
**mean imputation** and deal with these values.

```python
teams_stats['avg_shots'].hist();
```

We can see that most avg_shots values fall within the 7 - 14 range, so let's
fill in these entries with the average measured avg_shots

```python
shots_avg_team_avg = teams_stats['avg_shots'].mean()
# mean imputation
teams_stats.loc[(teams_stats['avg_shots'].isnull()), 'avg_shots'] = shots_avg_team_avg
# showing new values
teams_stats.describe()
```

```python
teams_stats.loc[(teams_stats['avg_shots'].isnull())]
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

```python
# possessions read, cleanup and merge
possessions_data = pd.read_csv("../dataset/possession_detail.csv")
last_possessions = possessions_data.sort_values(['elapsed'], ascending=False).drop_duplicates(subset=['match_id'])
last_possessions = last_possessions[['match_id', 'homepos', 'awaypos']]
```

After reading it, we need to see if the number of possession data we have
available is enough to be considered

```python
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

```python
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

As we can see, there are a lot of Nan values on the avg_corners column. This
represents teams that did not have corners data on this dataset. Instead of
removing thoose rows, and give less input to our models we need again to do
**mean imputation** and deal with these values.

```python
teams_stats['avg_corners'].hist();
```

We can see that most avg_corners values fall within the 8.5 - 12 range, so let's
fill in these entries with the average measured avg_corners

```python
corners_avg_team_avg = teams_stats['avg_corners'].mean()
# mean imputation
teams_stats.loc[(teams_stats['avg_corners'].isnull()), 'avg_corners'] = corners_avg_team_avg
# showing new values
teams_stats.describe()
```

Having done the **mean imputation** for team_attributes we can see that there
are no longer missing values for the avg_corners.

```python
teams_stats.loc[(teams_stats['avg_corners'].isnull())]
```

### Team Stats - Crosses
The final feature to be added is the crosses data. Normally the more dominant
team has more crosses because it creates more opportunity of goals during a
match. After merging all the data we need to watch for missing rows on the new
added column.

```python
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

As we can see, there are a lot of Nan values on the avg_crosses column. This
represents teams that did not have crosses data on this dataset. Instead of
removing thoose rows, and give less input to our models we need again to do
**mean imputation** and deal with these values

```python
teams_stats['avg_crosses'].hist();
```

We can see that most avg_crosses values fall within the 12.5 - 17.5 range, so
let's fill in these entries with the average measured avg_corners

```python
crosses_avg_team_avg = teams_stats['avg_crosses'].mean()
# mean imputation
teams_stats.loc[(teams_stats['avg_crosses'].isnull()), 'avg_crosses'] = crosses_avg_team_avg
# showing new values
teams_stats.describe()
```

Having done the **mean imputation** for team_attributes we can see that there
are no longer missing values for the avg_crosses.

```python
teams_stats.loc[(teams_stats['avg_crosses'].isnull())]
```

### FIFA data
We will now gather the fifa data regarding the overrall rating of the teams.
This will create some columns that will include the overall ratings of the
players that belong to a team. This way we can more easily compare the value of
each team.

```python
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

```python
fifa_data = get_fifa_data(viable_matches, player_attributes, None, data_exists = False)
fifa_data.describe()
```

### Joining all features
In this instance we need to join all features, select the input we will pass to
our models and drop the column regarding the outcome label.
To improve the overall measures of the supervised learning models we need to
normalize our features before training our models.

```python
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

```python
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

```python
clf = KNeighborsClassifier(n_neighbors=100)
train_predict(clf, features, outcomes)
```

### Decision Tree

```python
clf = DecisionTreeClassifier(random_state=0, criterion='entropy', splitter='random', max_depth=5)
train_predict(clf, features, outcomes)
```

### SVC

```python
clf = SVC(coef0=5, kernel='poly')
train_predict(clf, features, outcomes)
```

### Naive Bayes

```python
clf = GaussianNB(var_smoothing=1.1)
train_predict(clf, features, outcomes)
```

### Gradient Boosting

```python
clf = XGBClassifier(max_depth=20)
train_predict(clf, features, outcomes)
```

### Neural Network

```python
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

### Deep Neural Network

```python
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

```python

```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette("tab10")
from scipy import stats
from sklearn.model_selection import train_test_split
import os
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
from sklearn.metrics import explained_variance_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.linear_model import LinearRegression
seed = 1349
target = 'quality'

####################    ACQUIRE

def split_data(df):
    train_val,test = train_test_split(df,
                                     random_state=2013,
                                     train_size=0.82)
    train, validate = train_test_split(train_val,
                                      random_state=2013,
                                      train_size=0.73)
    return train, validate, test

def get_baseline(df):
    train, validate, test = split_data(df)
    train['baseline'] = 1
    baseline_accuracy = (train.baseline == train.playoffs).mean()
    subset = train[train.playoffs == 1]
    baseline_recall = (subset.baseline == subset.playoffs).mean()
    subset = train[train.baseline == 1]
    baseline_precision = (subset.baseline == subset.playoffs).mean()
    train.drop(columns='baseline',inplace=True)
    print(f'baseline accuracy: {baseline_accuracy:.2%}')
    print(f'baseline recall: {baseline_recall:.2%}')
    print(f'baseline precision: {baseline_precision:.2%}')  
    
    
def select_kbest(X, y, k):
    '''
    the function accepts the X_train data set, y_train array and k-number of features to select
    runs the SelectKBest algorithm and returns the list of features to be selected for the modeling
    !KBest doesn't depend on the model
    '''
    kbest = SelectKBest(f_regression, k=k)
    kbest.fit(X, y)
    return X.columns[kbest.get_support()].tolist()


def split_data(df):
    train_val,test = train_test_split(df,
                                     random_state=2013,
                                     train_size=0.82)
    train, validate = train_test_split(train_val,
                                      random_state=2013,
                                      train_size=0.73)
    return train, validate, test







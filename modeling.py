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

def run_model(X_train, X_validate, scaling):
    
    '''
    general function to run models with X_train and X_validate that were scaled
    '''

    for f in features:
        for key in models:
            # create a model
            model = models[key]
            # fit the model
            model.fit(X_train[features[f]], y_train)
            # predictions of the train set
            y_hat_train = model.predict(X_train[features[f]])
            # predictions of the validate set
            y_hat_validate = model.predict(X_validate[features[f]])
            # add train set predictions to the data frame
            predictions_train[key] = y_hat_train
            # add validate set predictions to the data frame
            predictions_validate[key] = y_hat_validate

            # calculate scores train set
            RMSE, R2 = regression_errors(y_train, y_hat_train)
            # calculate scores validation set
            RMSE_val, R2_val = regression_errors(y_validate, y_hat_validate)
            diff = np.abs(RMSE - RMSE_val)
            # calculate R2 difference
            R2_diff = R2 - R2_val
            
            # add the score results to the scores Data Frame
            scores.loc[len(scores.index)] = [key, f, scaling, RMSE, R2, RMSE_val, R2_val, diff, R2_diff]

def select_kbest(X, y, k):
    '''
    the function accepts the X_train data set, y_train array and k-number of features to select
    runs the SelectKBest algorithm and returns the list of features to be selected for the modeling
    !KBest doesn't depend on the model
    '''
    kbest = SelectKBest(f_regression, k=k)
    kbest.fit(X, y)
    return X.columns[kbest.get_support()].tolist()


models = {
    'Linear Regression': LinearRegression(),
    'Generalized Linear Model': TweedieRegressor(power=2, alpha = 0.5),
    'Gradient Boosting Regression': GradientBoostingRegressor(random_state=seed),
    'Decision Tree Regression': DecisionTreeRegressor(max_depth=4, random_state=seed),
    'Random Forest Regression':RandomForestRegressor(max_depth=4, random_state=seed),
    'LassoLars Regression':LassoLars(alpha=0.1)
    }


def full_split_wines(train, validate, test, target):
    '''
    accepts train, validate, test data sets and the name of the target variable as a parameter
    splits the data frame into:
    X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    #train, validate, test = train_validate_test_split(df, target)

    #save target column
    y_train = train[target]
    y_validate = validate[target]
    y_test = test[target]

    #remove target column from the sets
    train.drop(columns = target, inplace=True)
    validate.drop(columns = target, inplace=True)
    test.drop(columns = target, inplace=True)

    return train, validate, test, y_train, y_validate, y_test


X_train, X_validate, X_test, y_train, y_validate, y_test = full_split_wines(train, validate, test, target)


def standard_scale_wines(train, validate, test):
    '''
    accepts train, validate, test data sets
    scales the data in each of them
    returns transformed data sets
    '''

    col = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
    
    # create scalers
    scaler = StandardScaler()    
    #qt = QuantileTransformer(output_distribution='normal')
    scaler.fit(train[col])
    train[col] = scaler.transform(train[col])
    validate[col] = scaler.transform(validate[col])
    test[col] = scaler.transform(test[col])
    
    return train, validate, test

def run_model_standard():
    # runs regression models on the X_train scaled with StandardScaler()
    X1, X2, _ = standard_scale_wines(X_train, X_validate, X_test)
    run_model(X1, X2, 'standard')


X1, X2, X3 = standard_scale_wines(X_train, X_validate, X_test)


f1 = ['volatile acidity', 'chlorides', 'density']
f2 = ['volatile acidity', 'chlorides']
f3 = ['volatile acidity', 'chlorides', 'density', 'alcohol']
f4 = ['volatile acidity', 'chlorides', 'density', 'alcohol', 'residual sugar']
f5 = ['volatile acidity', 'chlorides', 'density', 'residual sugar', 'density', 'fixed acidity']
f6 = select_kbest(X_train, y_train, 4)
f7 = X_train.columns.tolist()

# create a dictionary with features
features = {
    'f1':f1,
    'f2':f2,
    'f3':f3,
    'f4':f4,
    'f5':f5,
    'f6':f6,
    'f7':f7
}



def regression_errors(y_actual, y_predicted):
    '''
    Calculates RMSE and R2 for regression models
    '''
    RMSE = np.sqrt(mean_squared_error(y_actual, y_predicted))
    R2 = r2_score(y_actual, y_predicted)
    return RMSE, R2


def run_best_model():
    '''
    the function runs the best model on the train, test and validate data sets 
    and returns scores in the data frame
    '''
    # create a data frame for test set results
    predictions_test = pd.DataFrame(y_test)
    predictions_test['baseline'] = baseline

    f = f7
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly.fit(X1[f])

    # create a df with transformed features of the train set
    X1_poly = pd.DataFrame(
                poly.transform(X1[f]),
                columns=poly.get_feature_names(X1[f].columns),
                index=X1.index)
    X1_poly = pd.concat([X1_poly, X1.iloc[:, 2:]], axis=1)

    # create a df with transformed features for the validate set
    X2_poly = pd.DataFrame(
                poly.transform(X2[f]),
                columns=poly.get_feature_names(X2[f].columns),
                index=X2.index)
    X2_poly = pd.concat([X2_poly, X2.iloc[:, 2:]], axis=1)

    # create a df with transformed features for the validate set
    X2_poly = pd.DataFrame(
                poly.transform(X2[f]),
                columns=poly.get_feature_names(X2[f].columns),
                index=X2.index)
    X2_poly = pd.concat([X2_poly, X2.iloc[:, 2:]], axis=1)

    # create. df with transformed features for the test set
    X3_poly = pd.DataFrame(
                poly.transform(X3[f]),
                columns=poly.get_feature_names(X3[f].columns),
                index=X3.index)
    X3_poly = pd.concat([X3_poly, X3.iloc[:, 2:]], axis=1)

    # create a Gradient Boosting Regression model
    model = GradientBoostingRegressor()
    # fit the model
    model.fit(X1_poly, y_train)
    # predictions of the train set
    y_hat_train = model.predict(X1_poly)
    # predictions of the validate set
    y_hat_validate = model.predict(X2_poly)
    # add train set predictions to the data frame
    y_hat_test = model.predict(X3_poly)
    predictions_test['predictions'] = y_hat_test

    # calculate scores train set
    RMSE_train, R2_train = regression_errors(y_train, y_hat_train)
    # calculate scores validation set
    RMSE_val, R2_val = regression_errors(y_validate, y_hat_validate)
    # calculate scores test set
    RMSE_test, R2_test = regression_errors(y_test, y_hat_test)
    RMSE_bl, _ = regression_errors(y_test, predictions_test.baseline)
    
    # save final score into a dictionary
    res = {
        'Features': str(f),
        'RMSE Train Set': RMSE_train,
        'RMSE Validation Set':RMSE_val,
        'RMSE Test Set':RMSE_test,
        'R2 Train Set':R2_train,
        'R2 Validation Set':R2_val,
        'R2 Test':R2_test,
        'Beats a basline by:':str(f'{round((RMSE_bl - RMSE_test) / RMSE_bl * 100, 1)}%')
    }

    # add the score results to the scores Data Frame
    final_test = pd.DataFrame({'Gradient Bosting Regression': list(res.keys()), 'Scores': list(res.values())})

    return final_test



def run_single():
    # create a list ['bedrooms', 'bathrooms', 'sq_feet', 'lot_sqft', 'house_age']
    single_corr = X1.iloc[:, :-3].columns.tolist()

    # for every single feature in the list
    for f in single_corr:
        # create a linear regression model
        model = LinearRegression()
        # fit the model
        model.fit(X1[[f]], y_train)
        # predictions of the train set
        y_hat_train = model.predict(X1[[f]])
        # predictions of the validate set
        y_hat_validate = model.predict(X2[[f]])
        # add train set predictions to the data frame
        predictions_train['single'] = y_hat_train
        # add validate set predictions to the data frame
        predictions_validate['single'] = y_hat_validate

        # calculate scores train set
        RMSE, R2 = regression_errors(y_train, y_hat_train)
        # calculate scores validation set
        RMSE_val, R2_val = regression_errors(y_validate, y_hat_validate)
        diff = np.abs(RMSE - RMSE_val)
        # calculate R2 difference
        R2_diff = R2 - R2_val
            
        # add the score results to the scores Data Frame
        scores.loc[len(scores.index)] = ['Single Linear Regression', f, 'standard', RMSE, R2, RMSE_val, R2_val, diff, R2_diff]


def run_polynomial():

    
    for i in range(1,5):
        # features[f] gives an access to the list of features in the dictionary
        #length = len(features[f])
        # create a Polynomial feature transformer
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        poly.fit(X1.iloc[:, :i])
        # create a df with transformed features of the train set
        X1_poly = pd.DataFrame(
            poly.transform(X1.iloc[:, :i]),
            columns=poly.get_feature_names(X1.iloc[:, :i].columns),
            index=X1.index)
        X1_poly = pd.concat([X1_poly, X1.iloc[:, i:]], axis=1)
        #X1_poly = pd.concat([X1_poly, X1], axis=1)
        
        #display(X1_poly.head(1)) #testing the columns
        
        # create a df with transformed features for the validate set
        X2_poly = pd.DataFrame(
            poly.transform(X2.iloc[:, :i]),
            columns=poly.get_feature_names(X2.iloc[:, :i].columns),
            index=X2.index)
        X2_poly = pd.concat([X2_poly, X2.iloc[:, i:]], axis=1)
        #X2_poly = pd.concat([X2_poly, X2], axis=1)
                             
        feature_name = 'poly'+str(i)
        
        for key in models:
            # create a model
            model = models[key]
            # fit the model
            model.fit(X1_poly, y_train)
            # predictions of the train set
            y_hat_train = model.predict(X1_poly)
            # predictions of the validate set
            y_hat_validate = model.predict(X2_poly)
            # add train set predictions to the data frame
            predictions_train[key] = y_hat_train
            # add validate set predictions to the data frame
            predictions_validate[key] = y_hat_validate

            # calculate scores train set
            RMSE, R2 = regression_errors(y_train, y_hat_train)
            # calculate scores validation set
            RMSE_val, R2_val = regression_errors(y_validate, y_hat_validate)
            diff = np.abs(RMSE - RMSE_val)
            # calculate R2 difference
            R2_diff = R2 - R2_val
            # add the score results to the scores Data Frame
            scores.loc[len(scores.index)] = [key, feature_name, 'standard', RMSE, R2, RMSE_val, R2_val, diff, R2_diff]


    def run_rfe():
     '''
    The function accepts the X_train data set, y_train array and k-number of features to select
    runs the RFE algorithm and returns the list of features to be selected for the modeling
    !RFE depends on the model.
    This function uses Linear regression
    '''
    # scale the data
    #X1, X2, _ = wr.standard_scale_zillow(X_train, X_validate, X_test)
    
    for key in models:
        # create a model
        model = models[key]
        
        # create a RFE feature selector
        rfe = RFE(model, n_features_to_select=4)
        rfe.fit(X1, y_train)
        
        # get the optimal features for every particular model
        f = X1.columns[rfe.get_support()].tolist()
        
        # fit the model with RFE features
        model.fit(X1[f], y_train)
        # predictions of the train set
        y_hat_train = model.predict(X1[f])
        # predictions of the validate set
        y_hat_validate = model.predict(X2[f])
        # add train set predictions to the data frame
        col_name = str(key)+'_rfe'
        predictions_train[col_name] = y_hat_train
        # add validate set predictions to the data frame
        predictions_validate[col_name] = y_hat_validate

        # calculate scores train set
        RMSE, R2 = regression_errors(y_train, y_hat_train)
        # calculate scores validation set
        RMSE_val, R2_val = regression_errors(y_validate, y_hat_validate)
        diff = np.abs(RMSE - RMSE_val)
        # calculate R2 difference
        R2_diff = R2 - R2_val
        # add the score results to the scores Data Frame
        scores.loc[len(scores.index)] = [key, 'rfe', 'standard', RMSE, R2, RMSE_val, R2_val, diff, R2_diff]


def scale_wines_quantile(train, validate, test):
    '''
    accepts train, validate, test data sets
    scales the data in each of them
    returns transformed data sets
    '''
    #count_columns = ['bedroomcnt', 'bathroomcnt']
    
    #col = train.columns[1:-1]
    col = ['volatile acidity', 'chlorides', 'density', 'residual sugar', 'density', 'fixed acidity']
    
    # create scalers
    #min_max_scaler = MinMaxScaler()    
    qt = QuantileTransformer(output_distribution='normal')
    qt.fit(train[col])
    train[col] = qt.transform(train[col])
    validate[col] = qt.transform(validate[col])
    test[col] = qt.transform(test[col])
    
    return train, validate, test

def run_model_quantile():
    XQ1, XQ2, _ = scale_wines_quantile(X_train, X_validate, X_test)
    run_model(XQ1, XQ2, 'quantile')


    def run_all_models():
     ''' the function runs all models and saves the results to csv file '''
    run_model_standard()
    run_model_quantile()
    run_rfe()
    run_polynomial()
    run_single()
    scores.to_csv('regression_results.csv')

def select_best_model_R2(scores):
    # select top 20 models based on the RMSE score of the train set
    top_20 = scores.sort_values(by='R2_train').head(20)
    # select top 5 models based on the RMSE score of the validate set
    top_5 = top_20.sort_values(by=['R2_validate']).head(5)
    # display top 5 models
    display(top_5)
    # select the best model with the smallest difference in the RMSE scores
    best_model = top_5.sort_values(by='R2_difference').head(1)
    return best_model


def select_best_model_RMSE(scores):
    # select top 20 models based on the RMSE score of the train set
    top_20 = scores.sort_values(by='RMSE_train').head(20)
    # select top 5 models based on the RMSE score of the validate set
    top_5 = top_20.sort_values(by=['RMSE_validate']).head(5)
    # display top 5 models
    display(top_5)
    # select the best model with the smallest difference in the RMSE scores
    best_model = top_5.sort_values(by='RMSE_difference').head(1)
    return best_model

def run_best_model():
    '''
    the function runs the best model on the train, test and validate data sets 
    and returns scores in the data frame
    '''
    # create a data frame for test set results
    predictions_test = pd.DataFrame(y_test)
    predictions_test['baseline'] = baseline

    f = f7
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly.fit(X1[f])

    # create a df with transformed features of the train set
    X1_poly = pd.DataFrame(
                poly.transform(X1[f]),
                columns=poly.get_feature_names(X1[f].columns),
                index=X1.index)
    X1_poly = pd.concat([X1_poly, X1.iloc[:, 2:]], axis=1)

    # create a df with transformed features for the validate set
    X2_poly = pd.DataFrame(
                poly.transform(X2[f]),
                columns=poly.get_feature_names(X2[f].columns),
                index=X2.index)
    X2_poly = pd.concat([X2_poly, X2.iloc[:, 2:]], axis=1)

    # create a df with transformed features for the validate set
    X2_poly = pd.DataFrame(
                poly.transform(X2[f]),
                columns=poly.get_feature_names(X2[f].columns),
                index=X2.index)
    X2_poly = pd.concat([X2_poly, X2.iloc[:, 2:]], axis=1)

    # create. df with transformed features for the test set
    X3_poly = pd.DataFrame(
                poly.transform(X3[f]),
                columns=poly.get_feature_names(X3[f].columns),
                index=X3.index)
    X3_poly = pd.concat([X3_poly, X3.iloc[:, 2:]], axis=1)

    # create a Gradient Boosting Regression model
    model = GradientBoostingRegressor()
    # fit the model
    model.fit(X1_poly, y_train)
    # predictions of the train set
    y_hat_train = model.predict(X1_poly)
    # predictions of the validate set
    y_hat_validate = model.predict(X2_poly)
    # add train set predictions to the data frame
    y_hat_test = model.predict(X3_poly)
    predictions_test['predictions'] = y_hat_test

    # calculate scores train set
    RMSE_train, R2_train = regression_errors(y_train, y_hat_train)
    # calculate scores validation set
    RMSE_val, R2_val = regression_errors(y_validate, y_hat_validate)
    # calculate scores test set
    RMSE_test, R2_test = regression_errors(y_test, y_hat_test)
    RMSE_bl, _ = regression_errors(y_test, predictions_test.baseline)
    
    # save final score into a dictionary
    res = {
        'Features': str(f),
        'RMSE Train Set': RMSE_train,
        'RMSE Validation Set':RMSE_val,
        'RMSE Test Set':RMSE_test,
        'R2 Train Set':R2_train,
        'R2 Validation Set':R2_val,
        'R2 Test':R2_test,
        'Beats a basline by:':str(f'{round((RMSE_bl - RMSE_test) / RMSE_bl * 100, 1)}%')
    }

    # add the score results to the scores Data Frame
    final_test = pd.DataFrame({'Gradient Bosting Regression': list(res.keys()), 'Scores': list(res.values())})

    return final_test
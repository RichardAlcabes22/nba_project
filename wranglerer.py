import pandas as pd
import numpy as np
import os
from env import user, pwd, host
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def acquire_red_df():
    ''' 
     Acquire data from X schema using env imports, rename columns, and storing a cached version of SQL pull as a .csv.
     Specifically, the SQL query returns a df in accordance with SQL script.
     ''' 
    if os.path.exists('winequality-red.csv'):
        print('local version found!')
        return pd.read_csv('winequality-red.csv')
    else:
        df = pd.read_csv('https://query.data.world/s/vca4kc5kqacfufuod2am7poidp5e4a?dws=00000')
        df.to_csv('winequality-red.csv',index=True)
        return df

def acquire_white_df():
    ''' 
     Acquire data from X schema using env imports, rename columns, and storing a cached version of SQL pull as a .csv.
     Specifically, the SQL query returns a df in accordance with SQL script.
     ''' 
    if os.path.exists('winequality-white.csv'):
        print('local version found!')
        return pd.read_csv('winequality-white.csv')
    else:
        df = pd.read_csv('https://query.data.world/s/42ad3ygx65hdnkyo7s4hhpillvmmxm?dws=00000')
        df.to_csv('winequality-white.csv',index=True)
        return df

def merge_wines_df():
    '''
    Takes in two dfs (red/white) and outputs a single merged df for both
    
    '''
    red,white = acquire_red_df(),acquire_white_df()
    red['wine_color'] = 1
    white['wine_color'] = 0
    df = pd.concat([red,white],axis=0,ignore_index=True)
    df.columns = [col.lower().replace(' ','_') for col in df.columns]
    return df


def concat_df():
    '''
    Takes in the vanilla red and white wine dataframes and returns a cleaned version that is ready
    for exploration and further analysis
    INPUT:
    NONE
    OUTPUT:
    wines = pandas dataframe with both red and white wine prepped for exploration
    '''
    red, white = acquire_red_df(),acquire_white_df()
    white['is_red'] = 0
    red['is_red'] = 1
    wines = pd.concat([red, white], ignore_index = True)
    return wines
    
    
def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    '''
    print('SUMMARY REPORT')
    print('=====================================================\n\n')
    print('Dataframe head: ')
    print(df.head(5))
    print('=====================================================\n\n')
    print('Dataframe info: ')
    print(df.info())
    print('=====================================================\n\n')
    print('Dataframe Description: ')
    print(df.describe())
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('=====================================================')
    print('DataFrame value counts: ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts(), '\n')
        else:
            print(df[col].value_counts(bins=10, sort=False), '\n')
    print('=====================================================')
    print('nulls in dataframe by column: ')
    print(nulls_by_col(df))
    print('=====================================================')
    print('nulls in dataframe by row: ')
    print(nulls_by_row(df))
    print('=====================================================')

    

def get_hist(df):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in df.columns if df[col].dtype != 'O']

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=50)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()
     
    

    
def get_upper_outliers(s, k=1.5):
    '''
    Given a series (ie df.col_name) and a cutoff value, k, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))


def add_upper_outlier_columns(df, k=1.5):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    for col in df.select_dtypes('number'):
        df[col + '_outliers_upper'] = get_upper_outliers(df[col], k)
    return df



    
def wrangle_student_math(path):
    df = pd.read_csv(path, sep=";")
    
    # drop any nulls
    df = df[~df.isnull()]

    # get object column names
    object_cols = get_object_cols(df)
    
    # create dummy vars
    df = create_dummies(df, object_cols)
      
    # split data 
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(df, 'G3')
    
    # get numeric column names
    numeric_cols = get_numeric_X_cols(X_train, object_cols)

    # scale data 
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(X_train, X_validate, X_test, numeric_cols)
    
    return df, X_train, X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test



def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

        
    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols
 
    
    
def create_dummies(df, object_cols):
    '''
    This function takes in a dataframe and list of object column names,
    and creates dummy variables of each of those columns. 
    It then appends the dummy variables to the original dataframe. 
    It returns the original df with the appended dummy variables. 
    '''
    
    # run pd.get_dummies() to create dummy vars for the object columns. 
    # we will drop the column representing the first unique value of each variable
    # we will opt to not create na columns for each variable with missing values 
    # (all missing values have been removed.)
    dummy_df = pd.get_dummies(object_cols, dummy_na=False, drop_first=True)
    
    # concatenate the dataframe with dummies to our original dataframe
    # via column (axis=1)
    df = pd.concat([df, dummy_df], axis=1)

    return df



def split_data(df):
    train_val,test = train_test_split(df,
                                     random_state=2013,
                                     train_size=0.7)
    train, validate = train_test_split(train_val,
                                      random_state=2013,
                                      train_size=0.8)
    return train, validate, test


def train_validate_test(df, target):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

        
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test



def get_numeric_X_cols(X_train, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]
    
    return numeric_cols




def nulls_by_col(df):
    '''
    This function takes in a dataframe 
    and finds the number of missing values in each column
    it returns a dataframe with quantity and percent of missing values
    '''
    num_missing = df.isnull().sum()
    percent_miss = num_missing / df.shape[0] * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': percent_miss})
    return cols_missing.sort_values(by='num_rows_missing', ascending=False)

def nulls_by_row(df):
    '''
    This function takes in a dataframe 
    and finds the number of missing values in a row
    it returns a new dataframe with quantity and percent of missing values
    '''
    num_missing = df.isnull().sum(axis=1)
    percent_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': percent_miss})
    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True)[['num_cols_missing', 'percent_cols_missing']]
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)

def remove_columns(df, cols_to_remove):
    '''
    This function takes in a dataframe 
    and the columns that need to be dropped
    then returns the desired dataframe.
    '''
    df = df.drop(columns=cols_to_remove)
    return df


def handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75):
    '''
    This function takes in a dataframe, the percent of columns and rows
    that need to have values/non-nulls
    and returns the dataframe with the desired amount of nulls left.
    '''
    column_threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=column_threshold)
    row_threshold = int(round(prop_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=row_threshold)
    return df

def data_prep(df, col_to_remove=[], prop_required_columns=0.5, prop_required_rows=0.75):
    '''
    This function uses two other functions to remove columns 
    and desired number of nulls values
    then returns the cleaned dataframe with acceptable number of nulls.
    '''
    df = remove_columns(df, col_to_remove)
    df = handle_missing_values(df, prop_required_columns, prop_required_rows)
    return df

def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    '''
    this function takes in 3 dataframes with the same columns, 
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler. 
    it returns 3 dataframes with the same column names and scaled values. 
    '''
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).


    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    #scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train. 
    # 
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, 
                                  columns=numeric_cols).\
                                  set_index([X_train.index.values])

    X_validate_scaled = pd.DataFrame(X_validate_scaled_array, 
                                     columns=numeric_cols).\
                                     set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, 
                                 columns=numeric_cols).\
                                 set_index([X_test.index.values])

    
    return X_train_scaled, X_validate_scaled, X_test_scaled



# Import libraries
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#################################

def df_convert_dtypes(df, convert_from, convert_to):
    '''Convert from one data type to another'''
    cols = df.select_dtypes(include=[convert_from]).columns
    for col in cols:
        df[col] = df[col].values.astype(convert_to)
    return df

#################################

def hist_text(ax, x_loc, y_loc, col, deci, size, full=True, title=''):
    '''Prints historgram statistics onto subplot.
    Adapted from https://github.com/lonnychen/necta-psle-dashboard.
    Parameters;
        ax (subplot): Matplotlib subplot to add text to
        x_loc (int, float): horizontal location to add text at
        y_loc (int, float): vertical location to add text at
        col (Series): DataFrame column to calculate statistics on
        deci (int): decimal places for rounding statistics
        size (string): Matplotlib text size string from 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'
        full (bool): prints all stats, else just basic ones
        title (string): Optional title
    Returns:
        None
    '''
    if full:
        ax.text(x_loc, y_loc, f'''{title}
        mean = {round(col.mean(), deci)}
        median = {round(col.median(), deci)}
        std = {round(col.std(), deci)}
        std/mean = {round(col.std()/col.mean(), deci)}
        skew = {round(col.skew(), deci)}
        kurtosis = {round(col.kurtosis(), deci)}''', size=size)
    else:
        ax.text(x_loc, y_loc, f'''{title}
        mean = {round(col.mean(), deci)}
        median = {round(col.median(), deci)}
        std = {round(col.std(), deci)}''', size=size)

#################################

def box_text(ax, x_loc, y_loc, col, deci, size, title=''):
    '''Prints box plot statistics onto subplot.
     Adapted from https://github.com/lonnychen/necta-psle-dashboard.
    Parameters;
        ax (subplot): Matplotlib subplot to add text to
        x_loc (int, float): horizontal location to add text at
        y_loc (int, float): vertical location to add text at
        col (Series): DataFrame column to calculate statistics on
        deci (int): decimal places for rounding values
        size (string): Matplotlib text size string from 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'
        title (string): Optional title
    Returns:
        None
    '''
    
    ax.text(x_loc, y_loc, f'''
    {title}
    {col.describe().round(deci)}''', size=size)

#################################
    
# Split the data into train and test based on specific time
def split_by_date(X, train_end_date):
    if not isinstance(train_end_date, pd.Timestamp):
        train_end_date = pd.Timestamp(train_end_date)

    # Convert index of X and y to Timestamp objects if they are strings
    if isinstance(X.index[0], str):
        X.index = pd.to_datetime(X.index)
        
    X_train = X[X.index <= train_end_date]
    X_test = X[X.index > train_end_date]
    return X_train, X_test


###############################

def prep_split(df, columns_drop, label, train_end_date, hours_ahead):
    # Add a column with the label shifted by "hours" rows

    if hours_ahead == 0:
        df['label_shifted'] = df[label]
    else:
        df['label_shifted'] = df[label].shift(-hours_ahead)
        # Drop the last "hours" rows as they have no label values
        df = df.iloc[:-hours_ahead]
   
    
    def split_by_date(X, train_end_date):
        if not isinstance(train_end_date, pd.Timestamp):
            train_end_date = pd.Timestamp(train_end_date)

        # Convert index of X and y to Timestamp objects if they are strings
        if isinstance(X.index[0], str):
            X.index = pd.to_datetime(X.index)

        X_train = X[X.index <= train_end_date]
        X_test = X[X.index > train_end_date]

        return X_train, X_test

    X = df.drop(columns=columns_drop + ['label_shifted'])
    y = df['label_shifted']

    X_train, X_test = split_by_date(X, train_end_date)
    y_train, y_test = split_by_date(y, train_end_date)

    # Standardize all columns except target
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)

    X_test_scaled = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

    return X_train, X_test, y_train, y_test


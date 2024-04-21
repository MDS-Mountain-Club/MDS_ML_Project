# Import libraries
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit


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

# Function to perform cross-validation 
'''
Fold cross-validation method might not be appropriate
as it could violate the temporal order of the data. 
Instead, we should use time series-specific methods like TimeSeriesSplit.
'''

def time_series_cv(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = {
        'RMSE': [], #I put here all metrics just in case, later we can remove some of them 
        'MSE': [],
        'MAPE': [],
        'MAE': []
    }
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)
        
        metrics['MSE'].append(mse)
        metrics['RMSE'].append(rmse)
        metrics['MAE'].append(mae)
        metrics['MAPE'].append(mape)
    
    return metrics



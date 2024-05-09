# Import libraries
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, PredictionErrorDisplay

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

###############################

def add_model_eval(model_eval_in, model_name, y_test, y_pred):
    '''Adds sklearn.metrics to a model evaluation dictionary'''
    
    model_eval_out = model_eval_in.copy() #else side effects!
    model_eval_out[model_name] = {'MSE': mean_squared_error(y_test, y_pred),
                                 'RMSE': root_mean_squared_error(y_test, y_pred),
                                 'R2': r2_score(y_test, y_pred),
                                 'MAE': mean_absolute_error(y_test, y_pred),
                                 'MAPE': mean_absolute_percentage_error(y_test, y_pred)}
    return model_eval_out

###############################

def model_eval_plot(y_true, y_pred, fig, axes, title=''):
    '''Plots key regression plots'''
    
    # scikit-learn provided plots!
    PredictionErrorDisplay.from_predictions(y_true=y_true, y_pred=y_pred, kind='actual_vs_predicted', ax=axes[0])
    PredictionErrorDisplay.from_predictions(y_true=y_true, y_pred=y_pred, kind='residual_vs_predicted', ax=axes[1])
    
    # Calculate residuals and plot on a histogram with text
    y_residuals = y_true - y_pred
    y_residuals.name = 'Residuals (actual - predicted)'
    bins = round(np.cbrt(len(y_residuals))) #cube root
    sns.histplot(x=y_residuals, bins=bins, ax=axes[2])
    x_loc = min(y_residuals)
    y_loc = np.sqrt(len(y_residuals))*5 #crude!
    hist_text(axes[2], x_loc, y_loc, y_residuals, 1, 'small')
    
    # Display niceties
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    
###############################
    
def regression_coef_plot(model, fig, axes, filter_val=0, title=''):
    '''Plots regression coefficients plot'''
    # Create sorted Series of coefficients and indices
    ols_coef_sorted = pd.Series(model.coef_, index=model.feature_names_in_).sort_values(ascending=False)
    ols_coef_sorted = ols_coef_sorted[abs(ols_coef_sorted) >= filter_val]
    
    # Plot and style
    sns.barplot(x=ols_coef_sorted.index, y=ols_coef_sorted.values, ax=axes)
    plt.title(title)
    plt.xticks(rotation=90, fontsize=8)
    plt.xlim([-1, len(ols_coef_sorted)])
    plt.xlabel('')

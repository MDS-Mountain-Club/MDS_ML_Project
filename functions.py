# Import libraries
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, PredictionErrorDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")


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


def split_by_date(X, train_end_date):
    '''Split the data into train and test based on specific time'''
    if not isinstance(train_end_date, pd.Timestamp):
        train_end_date = pd.Timestamp(train_end_date)

    # Convert index of X and y to Timestamp objects if they are strings
    if isinstance(X.index[0], str):
        X.index = pd.to_datetime(X.index)

    X_train = X[X.index <= train_end_date]
    X_test = X[X.index > train_end_date]
    return X_train, X_test


def split_by_date_and_standardize(X, train_end_date):
    '''Run `split_by_date` then add standardize both datasets.
       Returns DataFrames'''
    X_train, X_test = split_by_date(X, train_end_date)
    
    # Standardize X variables. 
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled

def prep_split(df, columns_drop, label, train_end_date, hours_ahead):
    '''Add a column with the label shifted by "hours" rows.
       No longer used.'''

    if hours_ahead == 0:
        df['label_shifted'] = df[label]
    else:
        df['label_shifted'] = df[label].shift(-hours_ahead)
        # Drop the last "hours" rows as they have no label values
        df = df.iloc[:-hours_ahead]

    X = df.drop(columns=columns_drop + ['label_shifted'])
    y = df['label_shifted']

    X_train, X_test = split_by_date(X, train_end_date)
    y_train, y_test = split_by_date(y, train_end_date)

    # Standardize all columns except target
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(
        X_train_scaled, index=X_train.index, columns=X_train.columns)

    X_test_scaled = scaler.transform(X_test)
    X_test = pd.DataFrame(
        X_test_scaled, index=X_test.index, columns=X_test.columns)

    return X_train, X_test, y_train, y_test

###############################


def add_model_eval(model_eval_in, model_name, y_test, y_pred):
    '''Adds sklearn.metrics to a model evaluation dictionary'''

    model_eval_out = model_eval_in.copy()  # else side effects!
    model_eval_out[model_name] = {'MSE': mean_squared_error(y_test, y_pred),
                                  'RMSE': root_mean_squared_error(y_test, y_pred),
                                  'R2': r2_score(y_test, y_pred),
                                  'MAE': mean_absolute_error(y_test, y_pred),
                                  'MAPE': mean_absolute_percentage_error(y_test, y_pred)}
    return model_eval_out

def model_eval_plot(y_true, y_pred, fig, axes, title=''):
    '''Plots key regression plots'''

    # scikit-learn provided plots!
    PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind='actual_vs_predicted', ax=axes[0])
    PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind='residual_vs_predicted', ax=axes[1])

    # Calculate residuals and plot on a histogram with text
    y_residuals = y_true - y_pred
    y_residuals.name = 'Residuals (actual - predicted)'
    bins = round(np.cbrt(len(y_residuals)))  # cube root
    sns.histplot(x=y_residuals, bins=bins, ax=axes[2])
    x_loc = min(y_residuals)
    y_loc = np.sqrt(len(y_residuals))*5  # crude!
    hist_text(axes[2], x_loc, y_loc, y_residuals, 1, 'small')

    # Display niceties
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    
def alpha_scores_cross_val(Model, X, y, cv, alphas, scoring='neg_mean_squared_error'):
    '''Helper function to get cross-validated alphas scores on a regression model'''
    
    # Setup alphas
    scores = defaultdict(list)
    scores['alphas'] = alphas
    
    # Alpha parameter loop
    for alpha in scores['alphas']:
        # Do CV with regression model
        model = Model(alpha=alpha)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        # Per-alpha: calculate scores average and SE
        scores['avg_mse'].append(-1 * np.mean(cv_scores))
        scores['std_errors'].append(np.std(cv_scores) / np.sqrt(len(cv_scores)))
        
    return scores

def best_alpha_one_se_rule(scores_dict):
    '''Helper function to find the best alpha using the One-Standard-Error rule'''
    scores_df = pd.DataFrame(scores_dict)
    within_one_std = scores_df['avg_mse'].min() + scores_df['std_errors'][scores_df['avg_mse'].idxmin()]
    best_alpha = max([scores_df['alphas'][i] for i, mse in enumerate(scores_df['avg_mse']) if mse <= within_one_std])
    return best_alpha, within_one_std

def plot_mse_vs_parameter(parameters, avg_mse, std_errors, best_param=None, within_one_std=None, log_scale=True, param_name='Alpha', model_name='Default Model Name'):
    '''Helper function to plot MSE vs. any varying parameter with standard errors,
       and optionally vertical and horizontal lines'''
    
    # Plotting MSE vs Parameter with standard error bars
    fig, axes = plt.subplots(figsize=(8, 4))
    plt.errorbar(parameters, avg_mse, yerr=std_errors, fmt='-o', ecolor='gray', capsize=5, capthick=2, label='MSE with Standard Errors')

    # Parameter and MSE lines for chosen point
    if best_param:
        plt.axvline(x=best_param, color='red', linestyle='--', label=f'Best {param_name}: {round(best_param,4)}')
    if within_one_std:
        plt.axhline(y=within_one_std, color='green', linestyle='--', label=f'Min MSE + 1 Std Error: {round(within_one_std,4)}')

    # Plot customizations
    plt.xlabel(param_name)
    plt.ylabel('Mean MSE')
    if log_scale:
        plt.xscale('log')  # Alpha values are on a logarithmic scale
    plt.title(f'MSE vs. {param_name} for {model_name}')
    plt.legend()
    plt.show()

def regression_coef_plot(model, fig, axes, filter_val=0, title=''):
    '''Plots regression coefficients plot'''
    # Create sorted Series of coefficients and indices
    ols_coef_sorted = pd.Series(
        model.coef_, index=model.feature_names_in_).sort_values(ascending=False)
    ols_coef_sorted = ols_coef_sorted[abs(ols_coef_sorted) >= filter_val]

    # Plot and style
    sns.barplot(x=ols_coef_sorted.index, y=ols_coef_sorted.values, ax=axes)
    plt.title(title)
    plt.xticks(rotation=90, fontsize=8)
    plt.xlim([-1, len(ols_coef_sorted)])
    plt.xlabel('')


###############################
    
    
def polynomial_terms(df_in, features_in, max_degree):
    '''Creates polynomial terms for input list of features and degrees'''
    df_out = df_in.copy()
    degrees = np.arange(2, max_degree+1, 1)
    for degree in degrees:
        for feature in features_in:
            feature_string = f'{feature}^{degree}'
            df_out[feature_string] = df_in[feature].apply(lambda x: x**degree)

    return df_out


###############################


def plot_true_pred(X_test, test_pred, test_true):
    '''Creates time-series plot of true and predicted values (y-axis) vs. time (x-axis)'''
    plt.figure(figsize=(10, 6))
    plt.plot(X_test.index, test_true, marker='o',
             color='blue', label='True Values')
    plt.plot(X_test.index, test_pred, marker='x', color='red',
             linestyle='--', label='Predicted Values')
    plt.title('True vs Predicted Values Over Time')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def split_array(X_arr, y_arr, n_hours):
    ''' Splits array by number of hours parameter.
       No longer used.'''
    X, y = list(), list()
    for window_start in range(len(X_arr)):
        past_end = window_start + n_hours
        if past_end + 1 > len(X_arr):
            break
        past, future = X_arr[window_start:past_end], y_arr[past_end]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)

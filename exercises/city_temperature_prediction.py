import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
MIN_TEMP = -20
BEST_K = 5


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to City Temp dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=["Date"])
    data = data.drop_duplicates().dropna()
    data = data[data.Temp > MIN_TEMP]
    data["DayOfYear"] = data["Date"].dt.dayofyear
    data["Year"] = data["Year"].astype(str)

    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("../datasets/City_Temperature.csv")
    y = pd.Series(X["Temp"])

    # Question 2 - Exploring data for specific country
    Israel_X = X[X.Country == "Israel"]
    px.scatter(Israel_X, x="DayOfYear", y="Temp", color="Year",
               title="Average Temp in Israel to Day of Year").show()

    px.bar(Israel_X.groupby(['Month'], as_index=False)
           .agg(std_dev=('Temp', 'std')), x='Month', y='std_dev',
           title="Standard deviation to Month in Israel").show()

    # Question 3 - Exploring differences between countries
    px.line(X.groupby(['Country', 'Month'], as_index=False)
            .agg(std_dev=('Temp', 'std'), mean=('Temp', 'mean')),
            x='Month', y='mean', error_y='std_dev', color='Country').show()

    # Question 4 - Fitting model for different values of `k`
    losses = []
    train_X_df, train_y_s, test_X_df, test_y_s = split_train_test(Israel_X.DayOfYear.to_frame(), Israel_X.Temp, 0.75)
    for k in range(1, 11):
        poly = PolynomialFitting(k)
        poly._fit(train_X_df.DayOfYear, train_y_s)
        losses.append(round(poly._loss(test_X_df.DayOfYear.to_numpy(), test_y_s.to_numpy()), 2))

    px.bar(pd.DataFrame({'x': [k for k in range(1, 11)], 'y': losses}), x="x", y="y",
               title=f"Loss as function of the degree of the polynom",
               labels={"x": f"Polynom Degree", "y": "Loss"}).show()

    # Question 5 - Evaluating fitted model on different countries

    train_X_df, train_y_s, test_X_df, test_y_s = split_train_test(Israel_X.DayOfYear.to_frame(), Israel_X.Temp, 0.75)
    Israel_m = PolynomialFitting(BEST_K)
    Israel_m._fit(train_X_df.DayOfYear, train_y_s)
    countries = list(set(X.Country))
    error_lst = []
    for country in countries:
        country_data = X[X.Country == country]
        error_lst.append(round(Israel_m._loss(country_data.DayOfYear.to_numpy(), country_data.Temp.to_numpy()), 2))

    px.bar(x=countries, y=error_lst, labels={'x': 'Country', 'y': 'Mean Square Error'}
           , title="MSE of temp prediction in each country with model fitted on Israel").show()
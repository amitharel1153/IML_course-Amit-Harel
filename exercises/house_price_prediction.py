from IMLearn.utils import split_train_test
from IMLearn.learners.regressors.linear_regression import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

train_proportion = 0.75
UNNECCERY_DATA = ["id", "date", "price", "lat", "long", "zipcode", "sqft_living15", "sqft_lot15"]
ALL_DATA = ["id", "date", "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
            "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built",
            "yr_renovated", "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"]

RELEVANT_DATA = [x for x in ALL_DATA if x not in UNNECCERY_DATA]


def preprocess_data(X: pd.DataFrame, y: pd.Series = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    if y is not None:
        y = y.dropna()
        y = y[y > 0]
        X = X.loc[y.index]
        X = X.dropna()
        y = y.loc[X.index]

    X = X.drop(columns=UNNECCERY_DATA)

    for col in list(X.columns):
        X[col] = X[col].fillna(0)

    if y is not None:
        return X, y
    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for col in X.columns:
        pearson_corr = round(np.cov(X[col], y)[0, 1] / (np.std(X[col]) * np.std(y)), 4)
        image = px.scatter(pd.DataFrame({'x': X[col], 'y': y}), x="x", y="y",
                           title=f"Response to {col} with Pearson Correlation {pearson_corr}",
                           labels={"x": f"{col}", "y": "Response"})

        image.show()
        # image.write_image(output_path + f"/{col}.png")  # TODO: doesnt work


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    train_X_df, train_y_s, test_X_df, test_y_s = split_train_test(df, df["price"], train_proportion)

    # Question 2 - Preprocessing of housing prices dataset
    train_X_df, train_y_s = preprocess_data(train_X_df, train_y_s)
    test_X_df = preprocess_data(test_X_df)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X_df, train_y_s, f"/images")

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    linear_reg = LinearRegression(True)
    p_results = None
    mean_loss_np = []
    std_dev_np = []
    p_lst = [p / 100 for p in range(10, 101)]
    for p in p_lst:
        p_results = []
        for i in range(10):
            train_X = train_X_df.sample(frac=p)
            train_y = train_y_s.loc[train_X.index]

            linear_reg._fit(train_X.to_numpy(), train_y.to_numpy())
            p_results.append(linear_reg._loss(test_X_df.to_numpy(), test_y_s.to_numpy()))

        mean_loss_np.append(np.mean(p_results))
        std_dev_np.append(np.std(p_results))

    mean_loss_np = np.array(mean_loss_np)
    std_dev_np = np.array(std_dev_np)

    go.Figure([
        go.Scatter(x=p_lst, y=mean_loss_np, mode='lines+markers', line=dict(color='blue'), name='Mean of MSE'),
        go.Scatter(x=p_lst, y=mean_loss_np - 2 * std_dev_np, mode='lines', line=dict(color='gray'), showlegend=False),
        go.Scatter(x=p_lst, y=mean_loss_np + 2 * std_dev_np, mode='lines', fill='tonexty', line=dict(color='gray'),
                   showlegend=False)],
        layout=go.Layout(title="MSE as function of training data size",
                         xaxis=dict(title='Percentage'), yaxis=dict(title='Mean Square Error'))).show()

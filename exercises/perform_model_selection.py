from __future__ import annotations
import numpy as np
import pandas as pd
import plotly
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    x, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, test_y = x[:n_samples], y[:n_samples], x[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_alphas, lasso_lamdas = np.linspace(1e-5, 0.05, num=n_evaluations), np.linspace(1e-3, 2, num=n_evaluations)
    ridge_scores, lasso_scores = np.zeros((n_evaluations, 2)), np.zeros((n_evaluations, 2))
    for i, (ra, la) in enumerate(zip(*[ridge_alphas, lasso_lamdas])):
        ridge_scores[i] = cross_validate(RidgeRegression(ra), train_X, train_y, mean_square_error, cv=5)
        lasso_scores[i] = cross_validate(Lasso(la, max_iter=3500), train_X, train_y, mean_square_error, cv=5)

    fig = make_subplots(1, 2, subplot_titles=["Ridge Regression", "Lasso Regression"],
                        shared_xaxes=True) \
        .update_layout(title="Train and Validation Errors:\n") \
        .update_xaxes(title="lambda - Regularization parameter") \
        .add_traces([go.Scatter(x=ridge_alphas, y=ridge_scores[:, 0], name="Ridge Train Error"),
                     go.Scatter(x=ridge_alphas, y=ridge_scores[:, 1], name="Ridge Validation Error"),
                     go.Scatter(x=lasso_lamdas, y=lasso_scores[:, 0], name="Lasso Train Error"),
                     go.Scatter(x=lasso_lamdas, y=lasso_scores[:, 1], name="Lasso Validation Error")],
                    rows=[1, 1, 1, 1],
                    cols=[1, 1, 2, 2])

    plotly.offline.offline.plot(fig)

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge_lambda = ridge_alphas[np.argmin(ridge_scores[:, 1])]
    best_lasso_lambda = lasso_lamdas[np.argmin(lasso_scores[:, 1])]

    fitted_ridge_loss = RidgeRegression(best_ridge_lambda).fit(train_X, train_y).loss(test_X, test_y)
    fitted_lasso_loss = mean_square_error(Lasso(best_lasso_lambda).fit(train_X, train_y).predict(test_X), test_y)
    linear_reg_loss = LinearRegression().fit(train_X, train_y).loss(test_X, test_y)

    print('Ridge Loss:', fitted_ridge_loss)
    print('Lasso Loss:', fitted_lasso_loss)
    print('Linear Regression Loss:', linear_reg_loss)


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()

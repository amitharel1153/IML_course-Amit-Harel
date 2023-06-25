import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test
from sklearn.metrics import roc_curve, auc
from utils import custom

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    weights = []
    values = []

    def update_data(solver, wgt, val, grad, t, eta, delta):
        weights.append(wgt)
        values.append(val)

    return update_data, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    data_to_plot_l1 = []
    data_to_plot_l2 = []
    for eta in etas:
        callback_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
        callback_l2, values_l2, weights_l2 = get_gd_state_recorder_callback()
        fixed_lr = FixedLR(eta)

        GradientDescent(learning_rate=fixed_lr, callback=callback_l1).fit(L1(np.copy(init)), None, None)
        GradientDescent(learning_rate=fixed_lr, callback=callback_l2).fit(L2(np.copy(init)), None, None)

        if eta == 0.01:
            plot_descent_path(module=L1, descent_path=np.array(weights_l1), title="L1 module descent path").show()
            plot_descent_path(module=L2, descent_path=np.array(weights_l2), title="L2 module descent path").show()

        norm_weights_l1 = [np.sqrt(np.sum(weights ** 2)) for weights in weights_l1]
        norm_weights_l2 = [np.sqrt(np.sum(weights ** 2)) for weights in weights_l2]

        data_to_plot_l1.append(
            go.Scatter(x=list(range(len(norm_weights_l1))), y=norm_weights_l1,
                       name=f"normalized weights L1, eta={eta}"))
        data_to_plot_l2.append(
            go.Scatter(x=list(range(len(norm_weights_l2))), y=norm_weights_l2,
                       name=f"normalized weights L2, eta={eta}"))

    go.Figure(
        data=data_to_plot_l1,
        layout=go.Layout(dict(
            title="Convergence rate L1",
            xaxis_title="Iteration",
            yaxis_title="Normalized Weights"))).show()

    go.Figure(
        data=data_to_plot_l2,
        layout=go.Layout(dict(
            title="Convergence rate L2",
            xaxis_title="Iteration",
            yaxis_title="Normalized Weights"))).show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    def check_alphas():
        alphas = np.linspace(0, 1, num=100)
        mis = []
        best_alpha, best_auc = None, 0
        for alpha in alphas:
            callback, values, weights = get_gd_state_recorder_callback()
            fitted_model = LogisticRegression(alpha=alpha, solver=GradientDescent(callback=callback)) \
                .fit(X_train.to_numpy(), y_train.to_numpy())
            mis.append(fitted_model.loss(X_test.to_numpy(), y_test.to_numpy()))

            fpr, tpr, thresholds = roc_curve(y_test.to_numpy(), fitted_model.predict_proba(X_test.to_numpy()))
            c = [custom[0], custom[-1]]
            if auc(fpr, tpr) > best_auc:
                best_auc = auc(fpr, tpr)
                best_alpha = alpha
                fig = go.Figure(
                    data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                                     name="Random Class Assignment"),
                          go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False,
                                     marker_size=5, marker_color=c[1][1],
                                     hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
                    layout=go.Layout(
                        title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}, alpha={alpha}$",
                        xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                        yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
        print(best_alpha)
        fig.show()

    # check_alphas()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    def choose_lamda():
        alpha = 0.5
        lamdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        max_iter = 20000
        lr = 1e-4
        n_evaluations = len(lamdas)
        l1_scores, l2_scores = np.zeros((n_evaluations, 2)), np.zeros((n_evaluations, 2))

        for i, lamda in enumerate(lamdas):
            l1_scores[i] = cross_validate(LogisticRegression(lam=lamda, alpha=alpha, penalty='l1',
                                                             solver=GradientDescent(learning_rate=FixedLR(lr),
                                                                                    max_iter=max_iter)),
                                          X_train.to_numpy(), y_train.to_numpy(), misclassification_error)

            l2_scores[i] = cross_validate(LogisticRegression(lam=lamda, alpha=alpha, penalty='l2',
                                                             solver=GradientDescent(learning_rate=FixedLR(lr),
                                                                                    max_iter=max_iter)),
                                          X_train.to_numpy(), y_train.to_numpy(), misclassification_error)

        best_lamda_l1 = lamdas[np.argmin(l1_scores[:, 1])]
        best_lamda_l2 = lamdas[np.argmin(l2_scores[:, 1])]

        l1_min_mis = np.min(l1_scores[:, 1])
        l2_min_mis = np.min(l2_scores[:, 1])

        print(f"The best lamda for l1 regression is: {best_lamda_l1}, with misclassification: {l1_min_mis}")
        print(f"The best lamda for l2 regression is: {best_lamda_l2}, with misclassification: {l2_min_mis}")

    # choose_lamda()


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()

import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):

    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train - and test errors of AdaBoost in noiseless case
    adaboost_model = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    train_error = [adaboost_model.partial_loss(train_X, train_y, t) for t in range(1, n_learners + 1)]
    test_error = [adaboost_model.partial_loss(test_X, test_y, t) for t in range(1, n_learners + 1)]

    plotly.offline.offline.plot(go.Figure(
        data=[go.Scatter(x=list(range(1, n_learners + 1)), y=train_error, name="Train error"),
              go.Scatter(x=list(range(1, n_learners + 1)), y=test_error, name="Test error")],
        layout=go.Layout(dict(
            title="AdaBoost Misclassification As Function Of Number Of Classifiers",
            xaxis_title="Iteration",
            yaxis_title="Misclassification Error"))))

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{t} Classifiers" for t in T])
    for i, t in enumerate(T):
        fig.add_traces(
            [decision_surface(lambda X: adaboost_model.partial_predict(X, t), lims[0], lims[1], density=60, showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                        marker=dict(color=test_y, symbol=np.where(test_y == 1, "circle", "x")))],
            rows=int(i/2)+1, cols=i % 2 + 1)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_t = np.argmin(test_error) + 1
    plotly.offline.offline.plot(go.Figure([
        decision_surface(lambda X: adaboost_model.partial_predict(X, best_t), lims[0], lims[1], density=60, showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                   marker=dict(color=test_y, symbol=np.where(test_y == 1, "circle", "x")))],
        layout=go.Layout(dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                title=f"Best Performing Ensemble Size: {best_t}, Accuracy: {1 - round(test_error[best_t - 1], 2)}"))))

    # Question 4: Decision surface with weighted samples
    D = adaboost_model.D_ / adaboost_model.D_.max() * 5
    plotly.offline.offline.plot(go.Figure([
        decision_surface(adaboost_model.predict, lims[0], lims[1], density=60, showscale=False),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                   marker=dict(size=D, color=train_y, symbol=np.where(train_y == 1, "circle", "x")))],
        layout=go.Layout(dict(
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    title=f"Final AdaBoost Sample Distribution"))))


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)


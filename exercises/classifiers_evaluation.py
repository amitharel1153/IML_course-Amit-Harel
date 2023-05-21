import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def my_callback(per: Perceptron, sample: np.ndarray, classif: int) -> None:
            losses.append(per.loss(X, y))

        perceptron = Perceptron(callback=my_callback)
        perceptron.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        go.Figure(go.Scatter(x=[x for x in range(1, len(losses) + 1)], y=losses, mode='lines'),
              layout=dict(title=f"misclassification error as function of number of iterations trained on ({n})",
                          xaxis_title='number of iterations trained on',
                          yaxis_title='misclassification error (normalized)')).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset('../datasets/' + f)

        # Fit models and predict over training set
        gaussian_obj = GaussianNaiveBayes().fit(X, y)
        lda_obj = LDA().fit(X, y)

        gaussian_prediction = gaussian_obj.predict(X)
        lda_prediction = lda_obj.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        # Add traces for data-points setting symbols and colors
        # X[:, 0] = first feature, X[:, 1] = second feature
        # To separate between the predictions I colored each class in different color
        gaussian_scatter = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                   marker=dict(color=gaussian_prediction, symbol=class_symbols[y]))
        lda_scatter = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                 marker=dict(color=lda_prediction, symbol=class_symbols[y]))

        fig = make_subplots(1, 2, subplot_titles=(f'Gaussian Naive Bayes, accuracy:{accuracy(y, gaussian_prediction)}',
                                                  f'LDA, accuracy:{accuracy(y, lda_prediction)}'))
        fig.add_traces([gaussian_scatter, lda_scatter], rows=[1, 1], cols=[1, 2])

        # Add `X` dots specifying fitted Gaussians' means
        # Draw the circle by the expectancy
        gaussian_x = go.Scatter(x=gaussian_obj.mu_[:, 0], y=gaussian_obj.mu_[:, 1], mode='markers',
                             marker=dict(color='black', symbol='x', size=10))
        lda_x = go.Scatter(x=lda_obj.mu_[:, 0], y=lda_obj.mu_[:, 1], mode='markers',
                           marker=dict(color='black', symbol='x', size=10))
        fig.add_traces([gaussian_x, lda_x], rows=[1, 1], cols=[1, 2])

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(gaussian_obj.classes_)):
            fig.add_traces([get_ellipse(gaussian_obj.mu_[i], np.diag(gaussian_obj.vars_[i])),
                            get_ellipse(lda_obj.mu_[i], lda_obj.cov_)], rows=[1, 1], cols=[1, 2])

        fig.update_layout(title_text=f'Gaussian Classifiers Comparison on dataset: {f}', showlegend=False)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    #compare_gaussian_classifiers()

from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] < self.threshold_, -self.sign_, self.sign_)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        threshold_vec_one = []
        miclassification_by_j_sign_one = []

        threshold_vec_minus = []
        miclassification_by_j_sign_minus = []

        for j in range(X.shape[1]):
            values = X[:, j]

            thr, mis = self._find_threshold(values, y, 1)
            threshold_vec_one.append(thr)
            miclassification_by_j_sign_one.append(mis)

            thr, mis = self._find_threshold(values, y, -1)
            threshold_vec_minus.append(thr)
            miclassification_by_j_sign_minus.append(mis)

        j_one = np.argmin(miclassification_by_j_sign_one)
        j_minus = np.argmin(miclassification_by_j_sign_minus)

        if miclassification_by_j_sign_one[j_one] > miclassification_by_j_sign_minus[j_minus]:
            self.sign_ = -1
            self.threshold_ = threshold_vec_minus[j_minus]
            self.j_ = j_minus

        else:
            self.sign_ = 1
            self.j_ = j_one
            self.threshold_ = threshold_vec_one[j_one]

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        err_minus_inf = np.sum(np.abs(labels)[np.sign(labels) != sign])

        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_labels = labels[sorted_indices]

        cumsum_values = np.cumsum(sorted_labels * sign)
        index_of_min = np.argmin(cumsum_values)

        err_of_value = err_minus_inf + cumsum_values[index_of_min]

        if err_minus_inf == np.min([err_of_value, err_minus_inf]):
            return -np.inf, float(err_minus_inf) / float(labels.size)
        elif err_of_value == np.min([err_of_value, err_minus_inf]):
            return sorted_values[index_of_min], float(err_of_value) / float(labels.size)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y_true=np.sign(y), y_pred=self._predict(X))

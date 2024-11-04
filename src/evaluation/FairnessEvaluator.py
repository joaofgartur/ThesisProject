"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""

import numpy as np
import pandas as pd

from utils import ratio, diff, abs_diff, conditional_probability, dict_to_dataframe
from constants import (PRIVILEGED, UNPRIVILEGED, POSITIVE_OUTCOME, NEGATIVE_OUTCOME,
                       TRUE_OUTCOME, PRED_OUTCOME)


class FairnessEvaluator(object):
    """
        Class to evaluate fairness metrics for a given dataset.

        Attributes
        ----------
        sensitive_attribute : str
            The name of the sensitive attribute.
        data : pd.DataFrame
            The combined dataset including features, sensitive attribute, targets, and predictions.

        Methods
        -------
        __init__(features: pd.DataFrame, targets: pd.DataFrame, predictions: pd.DataFrame, sensitive_attribute: pd.DataFrame):
            Initializes the FairnessEvaluator with the provided data.
        __get_counts(predicted_outcome, true_outcome, group_type, sensitive_attribute):
            Gets the counts of true and false values for a given group type and outcomes.
        __compute_false_error_balance_score(predicted_outcome, true_outcome):
            Computes the false error balance score for the given outcomes.
        __compute_rate_difference(true_outcome, pred_outcome):
            Computes the rate difference for the given true and predicted outcomes.
        disparate_impact():
            Computes the disparate impact metric.
        discrimination_score():
            Computes the discrimination score metric.
        false_positive_rate_diff():
            Computes the false positive rate difference.
        true_positive_rate_diff():
            Computes the true positive rate difference.
        consistency(k: int = 3):
            Computes the consistency metric.
        false_positive_error_rate_balance_score():
            Computes the false positive error rate balance score.
        false_negative_error_rate_balance_score():
            Computes the false negative error rate balance score.
        evaluate():
            Evaluates and returns all fairness metrics as a DataFrame.
        """

    def __init__(self, features: pd.DataFrame, targets: pd.DataFrame, predictions: pd.DataFrame,
                 sensitive_attribute: pd.DataFrame):
        """
        Initializes the FairnessEvaluator with the provided data.

        Parameters
        ----------
        features : pd.DataFrame
            The feature set of the dataset.
        targets : pd.DataFrame
            The target values of the dataset.
        predictions : pd.DataFrame
            The predicted values.
        sensitive_attribute : pd.DataFrame
            The sensitive attribute column.
        """

        x = features.drop(columns=[sensitive_attribute.columns[0]]) if sensitive_attribute.columns[0] in features \
            else features

        self.sensitive_attribute = sensitive_attribute.columns[0]
        self.data = pd.concat([x, sensitive_attribute, targets.squeeze().rename(TRUE_OUTCOME),
                               predictions.squeeze().rename(PRED_OUTCOME)], axis=1)

        del features

    def __get_counts(self, predicted_outcome, true_outcome, group_type, sensitive_attribute):
        """
        Gets the counts of true and false values for a given group type and outcomes.

        Parameters
        ----------
        predicted_outcome : int
            The predicted outcome value.
        true_outcome : int
            The true outcome value.
        group_type : int
            The group type (privileged or unprivileged).
        sensitive_attribute : str
            The name of the sensitive attribute.

        Returns
        -------
        tuple
            The counts of false and true values.
        """

        filtered_data = self.data[(self.data[PRED_OUTCOME] == predicted_outcome) &
                                  (self.data[TRUE_OUTCOME] == true_outcome) &
                                  (self.data[sensitive_attribute] == group_type)]
        f_v = filtered_data.shape[0]
        t_v = self.data[(self.data[PRED_OUTCOME] != predicted_outcome) &
                        (self.data[TRUE_OUTCOME] == true_outcome) &
                        (self.data[sensitive_attribute] == group_type)].shape[0]
        return f_v, t_v

    def __compute_false_error_balance_score(self, predicted_outcome, true_outcome):
        """
        Computes the false error balance score for the given outcomes.

        Parameters
        ----------
        predicted_outcome : int
            The predicted outcome value.
        true_outcome : int
            The true outcome value.

        Returns
        -------
        float
            The false error balance score.
        """

        f_v_u, t_v_u = self.__get_counts(predicted_outcome, true_outcome, UNPRIVILEGED, self.sensitive_attribute)
        f_v_p, t_v_p = self.__get_counts(predicted_outcome, true_outcome, PRIVILEGED, self.sensitive_attribute)

        return diff(1, abs_diff(ratio(f_v_u, (f_v_u + t_v_u)), ratio(f_v_p, (f_v_p + t_v_p))))

    def __compute_rate_difference(self, true_outcome, pred_outcome):
        """
        Computes the rate difference for the given true and predicted outcomes.

        Parameters
        ----------
        true_outcome : int
            The true outcome value.
        pred_outcome : int
            The predicted outcome value.

        Returns
        -------
        float
            The rate difference.
        """

        rate_privileged = conditional_probability(self.data, {PRED_OUTCOME: pred_outcome,
                                                              TRUE_OUTCOME: true_outcome,
                                                              self.sensitive_attribute: PRIVILEGED})

        rate_unprivileged = conditional_probability(self.data, {PRED_OUTCOME: pred_outcome,
                                                                TRUE_OUTCOME: true_outcome,
                                                                self.sensitive_attribute: UNPRIVILEGED})
        return diff(rate_privileged, rate_unprivileged)

    def disparate_impact(self):
        """
        Computes the disparate impact metric.

        Returns
        -------
        float
            The disparate impact.
        """

        unprivileged_cp = conditional_probability(self.data, {PRED_OUTCOME: POSITIVE_OUTCOME,
                                                              self.sensitive_attribute: UNPRIVILEGED})
        privileged_cp = conditional_probability(self.data, {PRED_OUTCOME: POSITIVE_OUTCOME,
                                                            self.sensitive_attribute: PRIVILEGED})
        return ratio(unprivileged_cp, privileged_cp)

    def discrimination_score(self):
        """
        Computes the discrimination score metric.

        Returns
        -------
        float
            The discrimination score.
        """

        unprivileged_cp = conditional_probability(self.data, {PRED_OUTCOME: POSITIVE_OUTCOME,
                                                              self.sensitive_attribute: UNPRIVILEGED})
        privileged_cp = conditional_probability(self.data, {PRED_OUTCOME: POSITIVE_OUTCOME,
                                                            self.sensitive_attribute: PRIVILEGED})
        return diff(1, abs_diff(unprivileged_cp, privileged_cp))

    def false_positive_rate_diff(self):
        """
        Computes the false positive rate difference.

        Returns
        -------
        float
            The false positive rate difference.
        """
        return abs_diff(1, self.__compute_rate_difference(NEGATIVE_OUTCOME, POSITIVE_OUTCOME))

    def true_positive_rate_diff(self):
        """
        Computes the true positive rate difference.

        Returns
        -------
        float
            The true positive rate difference.
        """
        return abs_diff(1, self.__compute_rate_difference(POSITIVE_OUTCOME, POSITIVE_OUTCOME))

    def consistency(self, k: int = 3):
        """
        Computes the consistency metric.

        Parameters
        ----------
        k : int, optional
            The number of neighbors to consider (default is 3).

        Returns
        -------
        float
            The consistency metric.
        """
        from sklearn.neighbors import NearestNeighbors
        data = self.data.to_numpy()

        x = np.ascontiguousarray(self.data.drop(columns=[TRUE_OUTCOME, PRED_OUTCOME]).to_numpy())
        y_pred = self.data.to_numpy()[:, -1]

        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
        nbrs.fit(x)
        indices = nbrs.kneighbors(x, return_distance=False)

        y_pred_knn = y_pred[indices]
        mean_y_pred_knn = np.mean(y_pred_knn, axis=1)

        consistency = diff(1, ratio(np.sum(np.abs(y_pred - mean_y_pred_knn)), data.shape[0]))

        return consistency

    def false_positive_error_rate_balance_score(self):
        """
        Computes the false positive error rate balance score.

        Returns
        -------
        float
            The false positive error rate balance score.
        """
        return self.__compute_false_error_balance_score(POSITIVE_OUTCOME, NEGATIVE_OUTCOME)

    def false_negative_error_rate_balance_score(self):
        """
        Computes the false negative error rate balance score.

        Returns
        -------
        float
            The false negative error rate balance score.
        """
        return self.__compute_false_error_balance_score(NEGATIVE_OUTCOME, POSITIVE_OUTCOME)

    def evaluate(self):
        """
        Evaluates and returns all fairness metrics as a DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing all the evaluated fairness metrics.
        """
        result = {
            'fairness_disparate_impact': self.disparate_impact(),
            'fairness_discrimination_score': self.discrimination_score(),
            'fairness_true_positive_rate_diff': self.true_positive_rate_diff(),
            'fairness_false_positive_rate_diff': self.false_positive_rate_diff(),
            'fairness_false_positive_error_rate_balance_score': self.false_positive_error_rate_balance_score(),
            'fairness_false_negative_error_rate_balance_score': self.false_negative_error_rate_balance_score(),
            'fairness_consistency': self.consistency(k=3),
        }

        return dict_to_dataframe(result)

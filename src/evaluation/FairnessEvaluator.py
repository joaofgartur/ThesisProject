"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from helpers import ratio, diff, abs_diff, conditional_probability, dict_to_dataframe
from constants import (PRIVILEGED, UNPRIVILEGED, POSITIVE_OUTCOME, NEGATIVE_OUTCOME,
                       TRUE_OUTCOME, PRED_OUTCOME)


class FairnessEvaluator(object):

    def __init__(self, features: pd.DataFrame, targets: pd.DataFrame, predictions: pd.DataFrame,
                 sensitive_attribute: pd.DataFrame):

        x = features.drop(columns=[sensitive_attribute.columns[0]]) if sensitive_attribute.columns[0] in features \
            else features

        self.sensitive_attribute = sensitive_attribute.columns[0]
        self.data = pd.concat([x, sensitive_attribute, targets.squeeze().rename(TRUE_OUTCOME),
                               predictions.squeeze().rename(PRED_OUTCOME)], axis=1)

        del features

    def __get_counts(self, predicted_outcome, true_outcome, group_type, sensitive_attribute):
        filtered_data = self.data[(self.data[PRED_OUTCOME] == predicted_outcome) &
                                  (self.data[TRUE_OUTCOME] == true_outcome) &
                                  (self.data[sensitive_attribute] == group_type)]
        f_v = filtered_data.shape[0]
        t_v = self.data[(self.data[PRED_OUTCOME] != predicted_outcome) &
                        (self.data[TRUE_OUTCOME] == true_outcome) &
                        (self.data[sensitive_attribute] == group_type)].shape[0]
        return f_v, t_v

    def __compute_false_error_balance_score(self, predicted_outcome, true_outcome):
        f_v_u, t_v_u = self.__get_counts(predicted_outcome, true_outcome, UNPRIVILEGED, self.sensitive_attribute)
        f_v_p, t_v_p = self.__get_counts(predicted_outcome, true_outcome, PRIVILEGED, self.sensitive_attribute)

        return diff(1, abs_diff(ratio(f_v_u, (f_v_u + t_v_u)), ratio(f_v_p, (f_v_p + t_v_p))))

    def __compute_rate_difference(self, true_outcome, pred_outcome):
        rate_privileged = conditional_probability(self.data, {PRED_OUTCOME: pred_outcome,
                                                              TRUE_OUTCOME: true_outcome,
                                                              self.sensitive_attribute: PRIVILEGED})

        rate_unprivileged = conditional_probability(self.data, {PRED_OUTCOME: pred_outcome,
                                                                TRUE_OUTCOME: true_outcome,
                                                                self.sensitive_attribute: UNPRIVILEGED})
        return diff(rate_privileged, rate_unprivileged)

    def disparate_impact(self):

        unprivileged_cp = conditional_probability(self.data, {PRED_OUTCOME: POSITIVE_OUTCOME,
                                                              self.sensitive_attribute: UNPRIVILEGED})
        privileged_cp = conditional_probability(self.data, {PRED_OUTCOME: POSITIVE_OUTCOME,
                                                            self.sensitive_attribute: PRIVILEGED})
        return ratio(unprivileged_cp, privileged_cp)

    def discrimination_score(self):
        unprivileged_cp = conditional_probability(self.data, {PRED_OUTCOME: POSITIVE_OUTCOME,
                                                              self.sensitive_attribute: UNPRIVILEGED})
        privileged_cp = conditional_probability(self.data, {PRED_OUTCOME: POSITIVE_OUTCOME,
                                                            self.sensitive_attribute: PRIVILEGED})
        return diff(1, abs_diff(unprivileged_cp, privileged_cp))

    def false_positive_rate_diff(self):
        return abs_diff(1, self.__compute_rate_difference(NEGATIVE_OUTCOME, POSITIVE_OUTCOME))

    def true_positive_rate_diff(self):
        return abs_diff(1, self.__compute_rate_difference(POSITIVE_OUTCOME, POSITIVE_OUTCOME))

    def consistency(self, k: int = 3):
        data = self.data.to_numpy()

        x = data[:, :-2]
        y_pred = data[:, -1]

        model = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(x)
        neighbors = model.kneighbors(x, return_distance=False)[:, 1:]

        y_pred_knn = y_pred[neighbors]
        mean_y_pred_knn = np.mean(y_pred_knn, axis=1)

        return diff(1, ratio(np.sum(np.abs(y_pred - mean_y_pred_knn)), data.shape[0]))

    def false_positive_error_rate_balance_score(self):
        return self.__compute_false_error_balance_score(POSITIVE_OUTCOME, NEGATIVE_OUTCOME)

    def false_negative_error_rate_balance_score(self):
        return self.__compute_false_error_balance_score(NEGATIVE_OUTCOME, POSITIVE_OUTCOME)

    def evaluate(self):

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

"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors

from datasets import Dataset
from helpers import ratio, diff, abs_diff, conditional_probability, dict_to_dataframe
from constants import (PRIVILEGED, UNPRIVILEGED, POSITIVE_OUTCOME, NEGATIVE_OUTCOME,
                       TRUE_OUTCOME, PRED_OUTCOME, NUM_DECIMALS)


def to_dataframe(array, labels, stats: bool, metric_name='Value'):
    def compute_stats(_array, _labels):
        _array = np.array(_array, dtype=float)
        mean_value = np.round(np.mean(_array), decimals=NUM_DECIMALS)
        std_value = np.round(np.std(_array), decimals=NUM_DECIMALS)
        _array = np.append(_array, [mean_value, std_value])
        _labels = _labels + ['Average', 'Std']

        return _array, _labels

    if stats:
        array, labels = compute_stats(array, labels)

    label_series = pd.Series(labels, name='Label')
    value_series = pd.Series(array, name=metric_name)

    df = pd.concat([label_series, value_series], axis=1)

    return df


class FairnessEvaluator(object):

    def __init__(self, original_data: Dataset, predicted_data: Dataset, sensitive_attribute: str):

        if sensitive_attribute in original_data.features:
            x = original_data.features.drop(columns=[sensitive_attribute])
        else:
            x = original_data.features

        s = original_data.protected_features[sensitive_attribute]

        y = original_data.targets.squeeze().rename(TRUE_OUTCOME)
        y_pred = predicted_data.targets.squeeze().rename(PRED_OUTCOME)

        self.sensitive_attribute = sensitive_attribute
        self.data = pd.concat([x, s, y, y_pred], axis=1)

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
        return self.__compute_rate_difference(NEGATIVE_OUTCOME, POSITIVE_OUTCOME)

    def true_positive_rate_diff(self):
        return self.__compute_rate_difference(POSITIVE_OUTCOME, POSITIVE_OUTCOME)

    def consistency(self, k):

        data_array = self.data.to_numpy()

        model = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(data_array)
        neighbors = model.kneighbors(data_array, return_distance=False)[:, 1:]

        n = data_array.shape[0]
        sum_value = 0
        pred_outcome_index = self.data.columns.get_loc(PRED_OUTCOME)

        for i in range(n):
            for j in neighbors[i]:
                sum_value += abs(data_array[i][pred_outcome_index] - data_array[j][pred_outcome_index])

        return diff(1, ratio(sum_value, n * k))

    def false_positive_error_rate_balance_score(self):
        return self.__compute_false_error_balance_score(POSITIVE_OUTCOME, NEGATIVE_OUTCOME)

    def false_negative_error_rate_balance_score(self):
        return self.__compute_false_error_balance_score(NEGATIVE_OUTCOME, POSITIVE_OUTCOME)

    def evaluate(self):

        result = {
            'disparate_impact': self.disparate_impact(),
            'discrimination_score': self.discrimination_score(),
            'true_positive_rate_diff': self.true_positive_rate_diff(),
            'false_positive_rate_diff': self.false_positive_rate_diff(),
            'false_positive_error_rate_balance_score': self.false_positive_error_rate_balance_score(),
            'false_negative_error_rate_balance_score': self.false_negative_error_rate_balance_score(),
            'consistency': self.consistency(k=3),
        }

        return dict_to_dataframe(result)

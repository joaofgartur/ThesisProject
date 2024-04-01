"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""
import copy

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors

from datasets import Dataset
from helpers import ratio
from evaluation import joint_probability
from constants import (PRIVILEGED, UNPRIVILEGED, POSITIVE_OUTCOME, NEGATIVE_OUTCOME,
                       TRUE_OUTCOME, PRED_OUTCOME, NUM_DECIMALS)


class FairnessEvaluator(object):

    def __init__(self, original_data: Dataset, predicted_data: Dataset, sensitive_attribute: str):
        y = original_data.targets.squeeze().rename(TRUE_OUTCOME)
        y_pred = predicted_data.targets.squeeze().rename(PRED_OUTCOME)
        self.data = pd.concat([original_data.features, original_data.original_protected_features, y, y_pred], axis=1)
        self.sensitive_attribute = 'orig_' + sensitive_attribute

    def disparate_impact(self):
        unprivileged_cp = ratio(
            joint_probability(self.data, {PRED_OUTCOME: POSITIVE_OUTCOME, self.sensitive_attribute: UNPRIVILEGED}),
            joint_probability(self.data, {self.sensitive_attribute: UNPRIVILEGED}))

        privileged_cp = ratio(
            joint_probability(self.data, {PRED_OUTCOME: POSITIVE_OUTCOME, self.sensitive_attribute: PRIVILEGED}),
            joint_probability(self.data, {self.sensitive_attribute: PRIVILEGED}))

        disparate_impact = np.round_(ratio(unprivileged_cp, privileged_cp), decimals=NUM_DECIMALS)

        return disparate_impact

    def discrimination_score(self):
        unprivileged_cp = ratio(
            joint_probability(self.data, {PRED_OUTCOME: POSITIVE_OUTCOME, self.sensitive_attribute: UNPRIVILEGED}),
            joint_probability(self.data, {self.sensitive_attribute: UNPRIVILEGED}))

        privileged_cp = ratio(
            joint_probability(self.data, {PRED_OUTCOME: POSITIVE_OUTCOME, self.sensitive_attribute: PRIVILEGED}),
            joint_probability(self.data, {self.sensitive_attribute: PRIVILEGED}))

        discrimination_score = np.round_(1 - abs(unprivileged_cp - privileged_cp), decimals=NUM_DECIMALS)

        return discrimination_score

    def false_positive_rate_diff(self):
        fpr_privileged = ratio(
            joint_probability(self.data, {self.sensitive_attribute: PRIVILEGED, TRUE_OUTCOME: NEGATIVE_OUTCOME,
                                          PRED_OUTCOME: POSITIVE_OUTCOME}),
            joint_probability(self.data, {self.sensitive_attribute: PRIVILEGED, TRUE_OUTCOME: NEGATIVE_OUTCOME}))

        fpr_unprivileged = ratio(
            joint_probability(self.data, {self.sensitive_attribute: UNPRIVILEGED, TRUE_OUTCOME: NEGATIVE_OUTCOME,
                                          PRED_OUTCOME: POSITIVE_OUTCOME}),
            joint_probability(self.data, {self.sensitive_attribute: UNPRIVILEGED, TRUE_OUTCOME: NEGATIVE_OUTCOME}))

        return np.round_(fpr_privileged - fpr_unprivileged, decimals=NUM_DECIMALS)

    def true_positive_rate_diff(self):
        tpr_privileged = ratio(
            joint_probability(self.data,
                              {self.sensitive_attribute: PRIVILEGED, TRUE_OUTCOME: POSITIVE_OUTCOME,
                               PRED_OUTCOME: POSITIVE_OUTCOME}),
            joint_probability(self.data, {self.sensitive_attribute: PRIVILEGED, TRUE_OUTCOME: POSITIVE_OUTCOME}))

        tpr_unprivileged = ratio(
            joint_probability(self.data,
                              {self.sensitive_attribute: UNPRIVILEGED, TRUE_OUTCOME: POSITIVE_OUTCOME,
                               PRED_OUTCOME: POSITIVE_OUTCOME}),
            joint_probability(self.data, {self.sensitive_attribute: UNPRIVILEGED, TRUE_OUTCOME: POSITIVE_OUTCOME}))

        return np.round_(tpr_privileged - tpr_unprivileged, decimals=NUM_DECIMALS)

    def consistency(self, k):

        data_array = self.data.to_numpy()

        model = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(data_array)
        neighbors = model.kneighbors(data_array, return_distance=False)[:, 1:]

        n = data_array.shape[0]
        sum_value = 0
        pred_outcome_index = self.data.columns.get_loc(PRED_OUTCOME)

        for i in range(n):
            for j in neighbors[i]:
                sum_value += abs(data_array[i][pred_outcome_index] - data_array[j][pred_outcome_index])

        consistency_score = 1 - (sum_value / (n * k))
        return np.round_(consistency_score, decimals=NUM_DECIMALS)

    def false_positive_error_rate_balance_score(self):
        f_p_unprivileged = self.data[(self.data[PRED_OUTCOME] == POSITIVE_OUTCOME) &
                                     (self.data[TRUE_OUTCOME] == NEGATIVE_OUTCOME) &
                                     (self.data[self.sensitive_attribute] == UNPRIVILEGED)].shape[0]

        t_n_unprivileged = self.data[(self.data[PRED_OUTCOME] == NEGATIVE_OUTCOME) &
                                     (self.data[TRUE_OUTCOME] == NEGATIVE_OUTCOME) &
                                     (self.data[self.sensitive_attribute] == UNPRIVILEGED)].shape[0]

        f_p_privileged = self.data[(self.data[PRED_OUTCOME] == POSITIVE_OUTCOME) &
                                   (self.data[TRUE_OUTCOME] == NEGATIVE_OUTCOME) &
                                   (self.data[self.sensitive_attribute] == PRIVILEGED)].shape[0]

        t_n_privileged = self.data[(self.data[PRED_OUTCOME] == NEGATIVE_OUTCOME) &
                                   (self.data[TRUE_OUTCOME] == NEGATIVE_OUTCOME) &
                                   (self.data[self.sensitive_attribute] == PRIVILEGED)].shape[0]

        result = 1 - abs(ratio(f_p_unprivileged, (f_p_unprivileged + t_n_unprivileged) -
                               ratio(f_p_privileged, (f_p_privileged + t_n_privileged))))
        return np.round_(result, decimals=NUM_DECIMALS)

    def false_negative_error_rate_balance_score(self):
        f_n_unprivileged = self.data[(self.data[PRED_OUTCOME] == NEGATIVE_OUTCOME) &
                                     (self.data[TRUE_OUTCOME] == POSITIVE_OUTCOME) &
                                     (self.data[self.sensitive_attribute] == UNPRIVILEGED)].shape[0]

        t_p_unprivileged = self.data[(self.data[PRED_OUTCOME] == POSITIVE_OUTCOME) &
                                     (self.data[TRUE_OUTCOME] == POSITIVE_OUTCOME) &
                                     (self.data[self.sensitive_attribute] == UNPRIVILEGED)].shape[0]

        f_n_privileged = self.data[(self.data[PRED_OUTCOME] == NEGATIVE_OUTCOME) &
                                   (self.data[TRUE_OUTCOME] == POSITIVE_OUTCOME) &
                                   (self.data[self.sensitive_attribute] == PRIVILEGED)].shape[0]

        t_p_privileged = self.data[(self.data[PRED_OUTCOME] == POSITIVE_OUTCOME) &
                                   (self.data[TRUE_OUTCOME] == POSITIVE_OUTCOME) &
                                   (self.data[self.sensitive_attribute] == PRIVILEGED)].shape[0]

        result = 1 - abs(ratio(f_n_unprivileged, (f_n_unprivileged + t_p_unprivileged)) -
                         ratio(f_n_privileged, (f_n_privileged + t_p_privileged)))
        return np.round_(result, decimals=NUM_DECIMALS)

    def evaluate(self):
        return {
            'disparate_impact': self.disparate_impact(),
            'discrimination_score': self.discrimination_score(),
            'true_positive_rate_diff': self.true_positive_rate_diff(),
            'false_positive_rate_diff': self.false_positive_rate_diff(),
            'consistency': self.consistency(5),
            'fperbs': self.false_positive_error_rate_balance_score(),
            'fnerbs': self.false_negative_error_rate_balance_score()
        }

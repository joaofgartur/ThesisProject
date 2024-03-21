"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""

import numpy as np
import pandas as pd

from datasets import Dataset
from helpers import safe_division
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
        unprivileged_cp = safe_division(
            joint_probability(self.data, {PRED_OUTCOME: POSITIVE_OUTCOME, self.sensitive_attribute: UNPRIVILEGED}),
            joint_probability(self.data, {self.sensitive_attribute: UNPRIVILEGED}))

        privileged_cp = safe_division(
            joint_probability(self.data, {PRED_OUTCOME: POSITIVE_OUTCOME, self.sensitive_attribute: PRIVILEGED}),
            joint_probability(self.data, {self.sensitive_attribute: PRIVILEGED}))

        try:
            return np.round_(safe_division(unprivileged_cp, privileged_cp), decimals=NUM_DECIMALS)
        except ZeroDivisionError:
            return 0.0

    def discrimination_score(self):
        unprivileged_cp = safe_division(
            joint_probability(self.data, {PRED_OUTCOME: POSITIVE_OUTCOME, self.sensitive_attribute: UNPRIVILEGED}),
            joint_probability(self.data, {self.sensitive_attribute: UNPRIVILEGED}))

        privileged_cp = safe_division(
            joint_probability(self.data, {PRED_OUTCOME: POSITIVE_OUTCOME, self.sensitive_attribute: PRIVILEGED}),
            joint_probability(self.data, {self.sensitive_attribute: PRIVILEGED}))

        return np.round_(unprivileged_cp - privileged_cp, decimals=NUM_DECIMALS)

    def false_positive_rate_diff(self):
        fpr_privileged = safe_division(
            joint_probability(self.data, {self.sensitive_attribute: PRIVILEGED, TRUE_OUTCOME: NEGATIVE_OUTCOME,
                                          PRED_OUTCOME: POSITIVE_OUTCOME}),
            joint_probability(self.data, {self.sensitive_attribute: PRIVILEGED, TRUE_OUTCOME: NEGATIVE_OUTCOME}))

        fpr_unprivileged = safe_division(
            joint_probability(self.data, {self.sensitive_attribute: UNPRIVILEGED, TRUE_OUTCOME: NEGATIVE_OUTCOME,
                                          PRED_OUTCOME: POSITIVE_OUTCOME}),
            joint_probability(self.data, {self.sensitive_attribute: UNPRIVILEGED, TRUE_OUTCOME: NEGATIVE_OUTCOME}))

        return np.round_(fpr_privileged - fpr_unprivileged, decimals=NUM_DECIMALS)

    def true_positive_rate_diff(self):
        tpr_privileged = safe_division(
            joint_probability(self.data,
                              {self.sensitive_attribute: PRIVILEGED, TRUE_OUTCOME: POSITIVE_OUTCOME,
                               PRED_OUTCOME: POSITIVE_OUTCOME}),
            joint_probability(self.data, {self.sensitive_attribute: PRIVILEGED, TRUE_OUTCOME: POSITIVE_OUTCOME}))

        tpr_unprivileged = safe_division(
            joint_probability(self.data,
                              {self.sensitive_attribute: UNPRIVILEGED, TRUE_OUTCOME: POSITIVE_OUTCOME,
                               PRED_OUTCOME: POSITIVE_OUTCOME}),
            joint_probability(self.data, {self.sensitive_attribute: UNPRIVILEGED, TRUE_OUTCOME: POSITIVE_OUTCOME}))

        return np.round_(tpr_privileged - tpr_unprivileged, decimals=NUM_DECIMALS)

    def evaluate(self):
        return {
            'disparate_impact': self.disparate_impact(),
            'discrimination_score': self.discrimination_score(),
            'true_positive_rate_diff': self.true_positive_rate_diff(),
            'false_positive_rate_diff': self.false_positive_rate_diff()
        }
"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""

import numpy as np
import pandas as pd

from datasets import Dataset
from helpers import safe_division
from metrics import conditional_probability, joint_probability
from constants import PRIVILEGED, UNPRIVILEGED, POSITIVE_OUTCOME, NEGATIVE_OUTCOME

Y = 'y'
Y_PRED = 'y_pred'


def disparate_impact(data: pd.DataFrame, sensitive_attribute: str):
    """
    Compute the disparate impact of a given label with respect to a sensitive attribute.

    Parameters
    ----------
    data :
        The DataFrame containing the dataset.
    sensitive_attribute :
        The sensitive attribute label.

    Returns
    -------
    float
        The disparate impact.
    """

    unprivileged_cp = safe_division(
        joint_probability(data, {Y_PRED: POSITIVE_OUTCOME, sensitive_attribute: UNPRIVILEGED}),
        joint_probability(data, {sensitive_attribute: UNPRIVILEGED}))

    privileged_cp = safe_division(
        joint_probability(data, {Y_PRED: POSITIVE_OUTCOME, sensitive_attribute: PRIVILEGED}),
        joint_probability(data, {sensitive_attribute: PRIVILEGED}))

    try:
        return np.round_(safe_division(unprivileged_cp, privileged_cp), decimals=4)
    except ZeroDivisionError:
        return 0.0


def discrimination_score(data: pd.DataFrame, sensitive_attribute: str):
    """
    Compute the discrimination score of a given label with respect to a sensitive attribute.

    Parameters
    ----------
    data :
        The DataFrame containing the dataset.
    sensitive_attribute :
        The sensitive attribute label.

    Returns
    -------
    float
        The discrimination score.

    """

    unprivileged_cp = safe_division(
        joint_probability(data, {Y: POSITIVE_OUTCOME, sensitive_attribute: UNPRIVILEGED}),
        joint_probability(data, {sensitive_attribute: UNPRIVILEGED}))

    privileged_cp = safe_division(
        joint_probability(data, {Y: POSITIVE_OUTCOME, sensitive_attribute: PRIVILEGED}),
        joint_probability(data, {sensitive_attribute: PRIVILEGED}))

    return np.round_(unprivileged_cp - privileged_cp, decimals=4)


def false_positive_rate_diff(data: pd.DataFrame, sensitive_attribute: str):
    fpr_privileged = safe_division(
        joint_probability(data, {sensitive_attribute: PRIVILEGED, Y: NEGATIVE_OUTCOME, Y_PRED: POSITIVE_OUTCOME}),
        joint_probability(data, {sensitive_attribute: PRIVILEGED, Y: NEGATIVE_OUTCOME}))

    fpr_unprivileged = safe_division(
        joint_probability(data, {sensitive_attribute: UNPRIVILEGED, Y: NEGATIVE_OUTCOME, Y_PRED: POSITIVE_OUTCOME}),
        joint_probability(data, {sensitive_attribute: UNPRIVILEGED, Y: NEGATIVE_OUTCOME}))

    return np.round_(fpr_privileged - fpr_unprivileged, decimals=4)


def true_positive_rate_diff(data: pd.DataFrame, sensitive_attribute: str):
    tpr_privileged = safe_division(
        joint_probability(data,
                          {sensitive_attribute: PRIVILEGED, Y: POSITIVE_OUTCOME, Y_PRED: POSITIVE_OUTCOME}),
        joint_probability(data, {sensitive_attribute: PRIVILEGED, Y: POSITIVE_OUTCOME}))

    tpr_unprivileged = safe_division(
        joint_probability(data,
                          {sensitive_attribute: UNPRIVILEGED, Y: POSITIVE_OUTCOME, Y_PRED: POSITIVE_OUTCOME}),
        joint_probability(data, {sensitive_attribute: UNPRIVILEGED, Y: POSITIVE_OUTCOME}))

    return np.round_(tpr_privileged - tpr_unprivileged, decimals=4)


def compute_metrics_suite(original_data: Dataset, predicted_data: Dataset, sensitive_attribute: str) -> dict:
    """
    Compute fairness metrics suite for a given dataset and sensitive attribute.

    Parameters
    ----------
    predicted_data
    original_data :
        The dataset object containing features, targets, and sensitive attributes.
    sensitive_attribute :
        The sensitive attribute label.

    Returns
    -------
    dict
        A dictionary containing fairness metrics.

    Notes
    -------
    - The current implemented metrics are Disparate Impact and Discrimination Score.
    """

    # prepare data
    y = original_data.targets.squeeze().rename(Y)
    y_pred = predicted_data.targets.squeeze().rename(Y_PRED)
    data = pd.concat([original_data.features, original_data.original_protected_features, y, y_pred], axis='columns')

    # prepare sensitive_attribute
    sensitive_attribute = 'orig_' + sensitive_attribute

    tpr_score = true_positive_rate_diff(data, sensitive_attribute)
    fpr_score = false_positive_rate_diff(data, sensitive_attribute)
    di = disparate_impact(data, sensitive_attribute)
    ds = discrimination_score(data, sensitive_attribute)

    metrics = {"Disparate Impact": di, "Discrimination Score": ds, 'True Positive Rate Diff': tpr_score,
               'False Positive Rate Diff': fpr_score}

    return metrics

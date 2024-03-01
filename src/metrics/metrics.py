"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""

import numpy as np
import pandas as pd

from datasets import Dataset
from metrics import conditional_probability
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

    unprivileged_cp = conditional_probability(data, Y, POSITIVE_OUTCOME, sensitive_attribute, UNPRIVILEGED)
    privileged_cp = conditional_probability(data, Y, POSITIVE_OUTCOME, sensitive_attribute, PRIVILEGED)

    try:
        return np.round_(unprivileged_cp / privileged_cp, decimals=8)
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

    unprivileged_cp = conditional_probability(data, Y, POSITIVE_OUTCOME, sensitive_attribute,
                                              UNPRIVILEGED)
    privileged_cp = conditional_probability(data, Y, POSITIVE_OUTCOME, sensitive_attribute,
                                            PRIVILEGED)

    return np.round_(unprivileged_cp - privileged_cp, decimals=8)


def compute_metrics_suite(dataset: Dataset, sensitive_attribute: str) -> dict:
    """
    Compute fairness metrics suite for a given dataset and sensitive attribute.

    Parameters
    ----------
    dataset :
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
    y = dataset.targets.squeeze().rename(Y)
    data = pd.concat([dataset.features, dataset.original_protected_features, y], axis='columns')

    # prepare sensitive_attribute
    sensitive_attribute = 'orig_' + sensitive_attribute

    di = disparate_impact(data, sensitive_attribute)
    ds = discrimination_score(data, sensitive_attribute)

    metrics = {"Disparate Impact": di, "Discrimination Score": ds}

    return metrics

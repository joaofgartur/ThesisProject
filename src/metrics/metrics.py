"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""

import numpy as np
import pandas as pd

from datasets import Dataset
from metrics import conditional_probability
from constants import PRIVILEGED, UNPRIVILEGED, POSITIVE_OUTCOME


def disparate_impact(data: pd.DataFrame, label: str, sensitive_attribute: str):
    """
    Compute the disparate impact of a given label with respect to a sensitive attribute.

    Parameters
    ----------
    data :
        The DataFrame containing the dataset.
    label :
        The target variable label.
    sensitive_attribute :
        The sensitive attribute label.

    Returns
    -------
    float
        The disparate impact.
    """
    unprivileged_cp = conditional_probability(data, label, POSITIVE_OUTCOME, sensitive_attribute, UNPRIVILEGED)
    privileged_cp = conditional_probability(data, label, POSITIVE_OUTCOME, sensitive_attribute, PRIVILEGED)

    return np.round_(unprivileged_cp / privileged_cp, decimals=4)


def discrimination_score(data: pd.DataFrame, label: str, sensitive_attribute: str):
    """
    Compute the discrimination score of a given label with respect to a sensitive attribute.

    Parameters
    ----------
    data :
        The DataFrame containing the dataset.
    label :
        The target variable label.
    sensitive_attribute :
        The sensitive attribute label.

    Returns
    -------
    float
        The discrimination score.

    """
    unprivileged_cp = conditional_probability(data, label, POSITIVE_OUTCOME, sensitive_attribute,
                                              UNPRIVILEGED)
    privileged_cp = conditional_probability(data, label, POSITIVE_OUTCOME, sensitive_attribute,
                                            PRIVILEGED)

    return np.round_(unprivileged_cp - privileged_cp, decimals=4)


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
    data, outcome_label = dataset.merge_features_and_targets()

    di = disparate_impact(data, outcome_label, sensitive_attribute)
    ds = discrimination_score(data, outcome_label, sensitive_attribute)

    metrics = {"disparate_impact": di, "discrimination_score": ds}

    return metrics

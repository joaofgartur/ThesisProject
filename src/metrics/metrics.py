import numpy as np
import pandas as pd
from datasets import Dataset
from constants import PRIVILEGED, UNPRIVILEGED, POSITIVE_OUTCOME
from metrics import conditional_probability


def disparate_impact(data: pd.DataFrame, label: str, sensitive_attribute: str):
    unprivileged_cp = conditional_probability(data, label, POSITIVE_OUTCOME, sensitive_attribute, UNPRIVILEGED)
    privileged_cp = conditional_probability(data, label, POSITIVE_OUTCOME, sensitive_attribute, PRIVILEGED)

    return np.round_(unprivileged_cp / privileged_cp, decimals=4)


def discrimination_score(data: pd.DataFrame, label: str, sensitive_attribute: str):
    unprivileged_cp = conditional_probability(data, label, POSITIVE_OUTCOME, sensitive_attribute,
                                              UNPRIVILEGED)  # women in case of compas
    privileged_cp = conditional_probability(data, label, POSITIVE_OUTCOME, sensitive_attribute,
                                            PRIVILEGED)  # men in case of compas
    return np.round_(unprivileged_cp - privileged_cp, decimals=4)


def compute_metrics_suite(dataset: Dataset, sensitive_attribute: str) -> dict:
    data = pd.concat([dataset.features, dataset.targets], axis="columns")
    label = dataset.targets.columns[0]

    di = disparate_impact(data, label, sensitive_attribute)
    ds = discrimination_score(data, label, sensitive_attribute)

    metrics = {"disparate_impact": di, "discrimination_score": ds}

    return metrics

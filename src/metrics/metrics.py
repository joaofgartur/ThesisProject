import pandas as pd
from datasets import Dataset
from constants import PRIVILEGED, UNPRIVILEGED, POSITIVE_OUTCOME
from metrics import conditional_probability


def disparate_impact(dataset: Dataset):
    data = pd.concat([dataset.features, dataset.targets], axis="columns")
    label = dataset.targets.columns[0]

    result = {}
    for sensitive_attribute in dataset.sensitive_attributes:
        unprivileged_cp = conditional_probability(data, label, POSITIVE_OUTCOME, sensitive_attribute, UNPRIVILEGED)
        privileged_cp = conditional_probability(data, label, POSITIVE_OUTCOME, sensitive_attribute, PRIVILEGED)
        result.update({sensitive_attribute: [unprivileged_cp / privileged_cp, privileged_cp, unprivileged_cp]})

    return result

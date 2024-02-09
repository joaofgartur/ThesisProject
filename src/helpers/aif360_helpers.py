import copy

import pandas as pd
from aif360.datasets import StandardDataset
import numpy as np

from constants import POSITIVE_OUTCOME, PRIVILEGED
from datasets import Dataset


def convert_to_standard_dataset(dataset: Dataset, sensitive_attribute: str):
    dataframe, label_name = dataset.merge_features_and_targets()
    favorable_classes = [POSITIVE_OUTCOME]
    privileged_classes = [PRIVILEGED]
    features_to_keep = dataset.features.columns.tolist()
    aif_dataset = StandardDataset(df=dataframe,
                                  label_name=label_name,
                                  favorable_classes=favorable_classes,
                                  privileged_classes=[privileged_classes],
                                  features_to_keep=features_to_keep,
                                  protected_attribute_names=[sensitive_attribute])

    return aif_dataset


def split_dataset(dataset: StandardDataset, split_ratio: float, shuffle: bool):
    train, test = dataset.split([split_ratio], shuffle=shuffle)
    return train, test


def concatenate_ndarrays(a: np.ndarray, b: np.ndarray):
    return np.concatenate((a, b), axis=0)


def modify_dataset(dataset: Dataset, features: np.ndarray, labels: np.ndarray):
    new_dataset = copy.deepcopy(dataset)
    new_dataset.features = pd.DataFrame(features, columns=dataset.features.columns)
    new_dataset.targets = pd.DataFrame(labels, columns=dataset.targets.columns)

    return new_dataset
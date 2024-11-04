"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""

from aif360.datasets import StandardDataset
import pandas as pd

from constants import POSITIVE_OUTCOME, PRIVILEGED
from datasets import Dataset


def convert_to_standard_dataset(dataset: Dataset, sensitive_attribute: str):
    dataframe = pd.concat([dataset.features, dataset.targets], axis='columns')
    label_name = dataset.targets.columns[0]
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

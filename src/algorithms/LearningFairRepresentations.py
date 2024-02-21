from aif360.algorithms.preprocessing import LFR

from algorithms.Algorithm import Algorithm
from constants import POSITIVE_OUTCOME, NEGATIVE_OUTCOME
from datasets import Dataset
from helpers import (convert_to_standard_dataset, set_dataset_features_and_labels, logger)

def is_dataframe_uniform(df):
    """
    Check if all elements in the DataFrame are the same.

    Parameters:
    df (DataFrame): Input DataFrame.

    Returns:
    bool: True if all elements are the same, False otherwise.
    """
    first_element = df.iloc[0, 0]  # Get the value of the first element
    return (df == first_element).all().all()


class LearningFairRepresentations(Algorithm):

    def __init__(self):
        super().__init__()

    def repair(self, dataset: Dataset, sensitive_attribute: str) -> Dataset:
        logger.info(f"Repairing dataset {dataset.name} via {self.__class__.__name__}...")

        # convert dataset into aif360 dataset
        standard_dataset = convert_to_standard_dataset(dataset, sensitive_attribute)

        # define privileged and unprivileged group
        privileged_groups = [{sensitive_attribute: POSITIVE_OUTCOME}]
        unprivileged_groups = [{sensitive_attribute: NEGATIVE_OUTCOME}]

        # transform dataset
        transformer = LFR(unprivileged_groups=unprivileged_groups,
                          privileged_groups=privileged_groups,
                          k=10, Ax=0.1, Ay=1.0, Az=2.0,
                          verbose=1)
        transformed_dataset = transformer.fit_transform(standard_dataset, maxiter=10)

        # convert into regular dataset
        new_dataset = set_dataset_features_and_labels(dataset=dataset,
                                                      features=transformed_dataset.features,
                                                      labels=transformed_dataset.labels)

        logger.info(f"Dataset {dataset.name} repaired.")

        return new_dataset

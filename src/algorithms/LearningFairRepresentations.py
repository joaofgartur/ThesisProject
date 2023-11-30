from aif360.algorithms.preprocessing import LFR
from sklearn.preprocessing import StandardScaler

from algorithms.Algorithm import Algorithm
from constants import POSITIVE_OUTCOME, NEGATIVE_OUTCOME
from datasets import Dataset
from helpers import (convert_to_standard_dataset, split_dataset, concatenate_ndarrays, modify_dataset)


class LearningFairRepresentations(Algorithm):

    def __init__(self, learning_settings: dict):
        self.learning_settings = learning_settings

    def repair(self, dataset: Dataset, sensitive_attribute: str):
        standard_dataset = convert_to_standard_dataset(dataset, sensitive_attribute)

        scaler = StandardScaler()
        train_dataset, test_dataset = split_dataset(standard_dataset, self.learning_settings["train_size"])
        train_dataset.features = scaler.fit_transform(train_dataset.features)
        test_dataset.features = scaler.transform(test_dataset.features)

        privileged_groups = [{sensitive_attribute: POSITIVE_OUTCOME}]
        unprivileged_groups = [{sensitive_attribute: NEGATIVE_OUTCOME}]

        transformer = LFR(unprivileged_groups=unprivileged_groups,
                          privileged_groups=privileged_groups,
                          k=10, Ax=0.1, Ay=1.0, Az=2.0,
                          verbose=1
                          )

        transformer = transformer.fit(train_dataset, maxiter=5000, maxfun=5000)

        dataset_transf_train = transformer.transform(train_dataset)
        dataset_transf_test = transformer.transform(test_dataset)

        features = concatenate_ndarrays(dataset_transf_train.features, dataset_transf_test.features)
        labels = concatenate_ndarrays(dataset_transf_train.labels, dataset_transf_test.labels)
        new_dataset = modify_dataset(dataset, features, labels)

        return new_dataset

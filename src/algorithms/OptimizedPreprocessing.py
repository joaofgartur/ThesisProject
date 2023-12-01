import numpy as np

from aif360.algorithms.preprocessing import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from sklearn.preprocessing import StandardScaler

from helpers import convert_to_standard_dataset, split_dataset, concatenate_ndarrays, modify_dataset
from algorithms.Algorithm import Algorithm
from datasets import Dataset


class OptimizedPreprocessing(Algorithm):

    def __init__(self, learning_settings: dict, optimization_parameters: dict, features_to_keep: list):
        self.learning_settings = learning_settings
        self.optimization_parameters = optimization_parameters
        self.features_to_keep = features_to_keep

    def repair(self, dataset: Dataset, sensitive_attribute: str):
        dataset.features = dataset.features[self.features_to_keep]
        standard_dataset = convert_to_standard_dataset(dataset, sensitive_attribute)
        # standard_dataset = load_preproc_data_adult(['race'])

        train_dataset, test_dataset = split_dataset(standard_dataset, self.learning_settings["train_size"], shuffle=True)

        optimizer = OptimPreproc(OptTools, self.optimization_parameters)
        print("fitting...")
        optimizer = optimizer.fit(train_dataset)
        print("done fitting...")

        dataset_transf_train = optimizer.transform(train_dataset, transform_Y=True)
        dataset_transf_train = train_dataset.align_datasets(dataset_transf_train)

        features = concatenate_ndarrays(dataset_transf_train.features, test_dataset.features)
        labels = concatenate_ndarrays(dataset_transf_train.labels, test_dataset.labels)
        new_dataset = modify_dataset(dataset, features, labels)

        return new_dataset

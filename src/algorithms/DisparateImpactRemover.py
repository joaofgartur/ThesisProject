import numpy as np

from aif360.algorithms.preprocessing import DisparateImpactRemover as DIR
from aif360.metrics import BinaryLabelDatasetMetric
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from algorithms.Algorithm import Algorithm
from algorithms.algorithms import scale_dataset
from datasets import Dataset
from helpers import convert_to_standard_dataset, split_dataset, concatenate_ndarrays, set_dataset_features_and_labels, logger


class DisparateImpactRemover(Algorithm):

    def __init__(self, repair_level: float):
        super().__init__()
        self.repair_level = repair_level

    def repair(self, dataset: Dataset, sensitive_attribute: str) -> Dataset:
        logger.info(f"Repairing dataset {dataset.name} via DisparateImpactRemover...")

        # convert dataset into aif360 dataset
        standard_dataset = convert_to_standard_dataset(dataset, sensitive_attribute)

        # transform dataset
        transformer = DIR(repair_level=self.repair_level)
        transformed_dataset = transformer.fit_transform(standard_dataset)

        # convert into regular dataset
        new_dataset = set_dataset_features_and_labels(dataset, transformed_dataset.features, transformed_dataset.labels)

        logger.info(f"Dataset {dataset.name} repaired.")

        return new_dataset

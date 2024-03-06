import abc
import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

from constants import PRIVILEGED, UNPRIVILEGED
from helpers import logger, extract_value, extract_filename


class Dataset(metaclass=abc.ABCMeta):

    def __init__(self, dataset_info: dict):
        """Initializes class instance"""
        self.features_mapping = None
        self.targets_mapping = None
        self.instance_weights = None

        self.__parse_dataset_info(dataset_info)

        # load raw data into dataframe
        self.features, self.targets = self._load_dataset()

        # preprocess dataset
        self._preprocessing()

        # save original values for protected attributes
        self.original_protected_features = self.get_protected_features().add_prefix('orig_')

        logger.info(f'[{extract_filename(__file__)}] Loaded.')

    @abc.abstractmethod
    def _load_dataset(self):
        """Loads the dataset"""

    @abc.abstractmethod
    def _transform_protected_attributes(self):
        """"""

    def get_protected_features(self) -> pd.DataFrame:
        return self.features.loc[:, self.protected_features]

    def merge_features_and_targets(self) -> (pd.DataFrame, str):
        data = pd.concat([self.features, self.targets], axis='columns')
        outcome = self.targets.columns[0]

        return data, outcome

    def get_train_sample_weights(self, train_set: np.ndarray) -> np.ndarray:
        if self.instance_weights is None:
            raise ValueError('Instance weights are none.')

        indexes = self.features.index[self.features.isin(train_set).any(axis=1)]

        return self.instance_weights[indexes]

    def split(self, settings: dict):
        x_train, x_test, y_train, y_test = train_test_split(self.features,
                                                            self.targets,
                                                            train_size=settings['train_size'],
                                                            random_state=settings['seed'])

        split_ratio = settings['test_size'] / (settings['validation_size'] + settings['test_size'])
        x_val, x_test, y_val, y_test = train_test_split(x_test,
                                                        y_test,
                                                        test_size=split_ratio,
                                                        random_state=settings['seed'])

        train_set = update_dataset(self, features=x_train, targets=y_train)
        train_set.__reset_indexes()

        validation_set = update_dataset(self, features=x_val, targets=y_val)
        validation_set.__reset_indexes()

        test_set = update_dataset(self, features=x_test, targets=y_test)
        test_set.__reset_indexes()

        return train_set, validation_set, test_set

    def print_dataset(self):
        """Prints the dataset"""
        print(f'---- Features ----\n{self.features}')
        print(f'---- targets ----\n{self.targets}')

    def _preprocessing(self):
        logger.info("[DATASET] Pre-processing features and targets.")

        self.__drop_invalid_instances()
        self.__quantize_numerical_features()
        self._transform_protected_attributes()
        self.__one_hot_encode_categorical_features()
        self.__convert_categorical_targets()

        logger.info("[DATASET] Pre-processing complete.")

    def __drop_invalid_instances(self):
        drop_indexes = [index for index, row in self.features.iterrows() if row.isnull().any()]
        self.features = self.features.drop(drop_indexes).reset_index(drop=True)
        self.targets = self.targets.drop(drop_indexes).reset_index(drop=True)

    def __parse_dataset_info(self, dataset_info: dict):
        dataset_name = 'dataset_name'
        protected_attributes = 'protected_attributes'
        explanatory_attributes = 'explanatory_attributes'
        privileged_classes = 'privileged_classes'
        target = 'target'

        self.name = extract_value(dataset_name, dataset_info)
        self.protected_features = extract_value(protected_attributes, dataset_info)
        self.explanatory_features = extract_value(explanatory_attributes, dataset_info)
        self.privileged_classes = extract_value(privileged_classes, dataset_info)
        self.target = extract_value(target, dataset_info)

    def __reset_indexes(self):
        self.features = self.features.reset_index(drop=True)
        self.targets = self.targets.reset_index(drop=True)

    def __quantize_numerical_features(self, n_bins=4):
        numerical_data = self.features.select_dtypes(include=['number'])
        to_drop = []
        for attribute in self.protected_features:
            print(attribute)
            if attribute in numerical_data.columns:
                to_drop.append(attribute)
        numerical_data = numerical_data.drop(columns=to_drop)
        categorical_data = self.features.drop(numerical_data, axis=1)

        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        discretized_numerical_data = pd.DataFrame(discretizer.fit_transform(numerical_data),
                                                  columns=numerical_data.columns)
        processed_features = pd.concat([categorical_data, discretized_numerical_data], axis=1)

        self.features = processed_features

    def __one_hot_encode_categorical_features(self):
        numerical_data = self.features.select_dtypes(include=['number'])
        categorical_data = self.features.drop(numerical_data, axis=1)

        encoded_categorical_data = pd.get_dummies(categorical_data, columns=categorical_data.columns)
        encoded_categorical_data *= 1.0

        processed_features = pd.concat([numerical_data, encoded_categorical_data], axis=1)

        self.features = processed_features

    def __convert_categorical_targets(self):
        label_encoder = LabelEncoder()
        labels_mapping = {}

        for column in self.targets.columns:
            if self.targets[column].dtype == object:
                self.targets[column] = label_encoder.fit_transform(self.targets[column])
                mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
                labels_mapping.update({column: mapping})

        self.targets_mapping = labels_mapping


def update_dataset(dataset: Dataset, features: np.ndarray = None, targets: np.ndarray = None):
    def update_features():
        updated_dataset.features = pd.DataFrame(features, columns=dataset.features.columns)

    def update_targets():
        updated_dataset.targets = pd.DataFrame(targets, columns=dataset.targets.columns)

    updated_dataset = copy.deepcopy(dataset)

    if features is not None:
        update_features()

    if targets is not None:
        update_targets()

    return updated_dataset


def is_privileged(instance: str, privileged: str):
    return PRIVILEGED * 1.0 if instance == privileged else UNPRIVILEGED * 1.0

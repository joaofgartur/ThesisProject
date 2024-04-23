import abc
import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

from constants import PRIVILEGED, UNPRIVILEGED
from helpers import logger, extract_value, extract_filename


class Dataset(metaclass=abc.ABCMeta):

    def __init__(self, dataset_info: dict, seed: int):
        """Initializes class instance"""
        self.features_mapping = None
        self.targets_mapping = None
        self.instance_weights = None
        self.protected_attributes = None
        self.seed = seed

        self.__parse_dataset_info(dataset_info)

        # load raw data into dataframe
        self.features, self.targets = self._load_dataset()

        # preprocess dataset
        self._preprocessing()

        self.save_protected_attributes()

        logger.info(f'[{extract_filename(__file__)}] Loaded.')

    @abc.abstractmethod
    def _load_dataset(self):
        """Loads the dataset"""

    @abc.abstractmethod
    def _transform_protected_attributes(self):
        """"""

    def save_protected_attributes(self):
        self.protected_attributes = self.features.loc[:, self.protected_features_names]

    def set_feature(self, feature: str, series: pd.DataFrame):
        if feature not in self.features.columns:
            raise ValueError(f'Feature {feature} does not exist.')

        self.features[feature] = series

    def get_protected_attributes(self):
        return self.protected_attributes.loc[:, self.protected_features_names]

    def get_protected_feature(self, feature) -> pd.DataFrame:
        if feature not in self.protected_features_names:
            raise ValueError(f'Feature {feature} is not a protected feature.')

        return self.protected_attributes[feature]

    def get_dummy_protected_feature(self, feature) -> pd.DataFrame:
        if feature not in self.protected_features_names:
            raise ValueError(f'Attribute {feature} is not a protected attribute.')

        protected_feature = self.get_protected_feature(feature)
        dummy_protected_feature = pd.get_dummies(protected_feature, dtype=float)
        dummy_protected_feature = dummy_protected_feature.rename(columns=self.features_mapping[feature])

        return dummy_protected_feature

    def get_value_counts(self, feature):
        df = self.features[feature]
        df.replace(self.features_mapping[feature], inplace=True)
        return df.value_counts()

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

        # split into train and validation/test sets
        stratify_criteria = pd.concat([self.features[self.protected_features_names], self.targets], axis=1)

        # Assuming 'subgroup_column' contains the subgroup identifiers
        subgroup_counts = stratify_criteria.value_counts()

        print(subgroup_counts)

        # Identify subgroups with dimensionality 1
        subgroups_to_remove = subgroup_counts[subgroup_counts == 1].index

        print(subgroups_to_remove)

        indexes_to_remove = []
        columns = subgroups_to_remove.names
        for i in range(len(subgroups_to_remove)):
            df = copy.deepcopy(stratify_criteria)
            for j in range(len(columns)):
                df = df[df[columns[j]] == subgroups_to_remove[i][j]]

            indexes_to_remove.append(df.index.to_list()[0])
            
        print(f'to remove:\n {indexes_to_remove}')

        print(self.features.shape)
        self.features = self.features.drop(index=indexes_to_remove)
        print(self.features.shape)

        print(self.targets.shape)
        self.targets = self.targets.drop(index=indexes_to_remove)
        print(self.targets.shape)

        print(stratify_criteria.value_counts())
        stratify_criteria = stratify_criteria.drop(index=indexes_to_remove)
        print(stratify_criteria.value_counts())


        x_train, x_test, y_train, y_test = train_test_split(self.features, self.targets,
                                                            train_size=settings.get('train_size'),
                                                            random_state=self.seed,
                                                            shuffle=True,
                                                            stratify=stratify_criteria)

        # split into validation and test sets
        split_ratio = settings['test_size'] / (settings['validation_size'] + settings['test_size'])
        stratify_criteria = pd.concat([x_test[self.protected_features_names], y_test], axis=1)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                        test_size=split_ratio,
                                                        random_state=self.seed,
                                                        shuffle=True,
                                                        stratify=stratify_criteria)

        train_set = update_dataset(self, features=x_train, targets=y_train)
        train_set.__reset_indexes()
        train_set.save_protected_attributes()

        validation_set = update_dataset(self, features=x_val, targets=y_val)
        validation_set.__reset_indexes()
        validation_set.save_protected_attributes()

        test_set = update_dataset(self, features=x_test, targets=y_test)
        test_set.__reset_indexes()
        test_set.save_protected_attributes()

        return train_set, validation_set, test_set

    def print_dataset(self):
        """Prints the dataset"""
        print(f'---- Features ----\n{self.features}')
        print(f'---- targets ----\n{self.targets}')

    def _preprocessing(self):
        logger.info("[DATASET] Pre-processing features and targets.")

        if 'Aif' not in self.name:
            self.__drop_invalid_instances()
            self.__quantize_numerical_features()
            self._transform_protected_attributes()
            # self.__one_hot_encode_categorical_features()
            self.__label_encode_categorical(features=True, targets=True)

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
        self.protected_features_names = extract_value(protected_attributes, dataset_info)
        self.explanatory_features = extract_value(explanatory_attributes, dataset_info)
        self.privileged_classes = extract_value(privileged_classes, dataset_info)
        self.target = extract_value(target, dataset_info)

    def __reset_indexes(self):
        self.features = self.features.reset_index(drop=True)
        self.targets = self.targets.reset_index(drop=True)

    def __quantize_numerical_features(self, n_bins=4):
        numerical_data = self.features.select_dtypes(include=['number'])
        to_drop = []
        for attribute in self.protected_features_names:
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

        if categorical_data.shape[1]:
            encoded_categorical_data = pd.get_dummies(categorical_data, columns=categorical_data.columns)
            encoded_categorical_data *= 1.0

            processed_features = pd.concat([numerical_data, encoded_categorical_data], axis=1)

            self.features = processed_features

    def __label_encode_categorical(self, features=False, targets=True):

        def __label_encode(df: pd.DataFrame):
            encoder = LabelEncoder()
            mapping = {}

            for column in df.columns:
                if df[column].dtype == object:
                    df[column] = encoder.fit_transform(df[column])
                    local_mapping = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
                    mapping.update({column: local_mapping})

            return df, mapping

        if features:
            self.features, self.features_mapping = __label_encode(self.features)

        if targets:
            self.targets, self.targets_mapping = __label_encode(self.targets)


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


def match_features(dataset_a: Dataset, dataset_b: Dataset):
    df = copy.deepcopy(dataset_b)
    common_columns = dataset_a.features.columns.intersection(dataset_b.features.columns)
    df.features = dataset_b.features.drop(columns=dataset_b.features.columns.difference(common_columns))

    return df

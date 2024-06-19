import abc
import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

from helpers import logger, extract_filename, get_seed


class DatasetConfig:
    """
    Configuration class for a Dataset.

    Attributes
    ----------
    name : str
        The name of the dataset.
    protected_attributes : list[str]
        The list of protected attributes in the dataset.
    target : str
        The target attribute in the dataset.
    train_size : float
        The proportion of the dataset to include in the train split.
    validation_size : float
        The proportion of the dataset to include in the validation split.
    test_size : float
        The proportion of the dataset to include in the test split.
    """

    def __init__(self, name: str, protected_attributes: list[str], target: str,
                 train_size: float, test_size: float, validation_size: float):
        """
        Initializes the DatasetConfig object with the provided dataset information.

        Parameters
        ----------
        name : str
            The name of the dataset.
        protected_attributes : list[str]
            The list of protected attributes in the dataset.
        target : str
            The target attribute in the dataset.
        train_size : float
            The proportion of the dataset to include in the train split.
        validation_size : float
            The proportion of the dataset to include in the validation split.
        test_size : float
            The proportion of the dataset to include in the test split.
        """
        self.name = name
        self.protected_attributes = protected_attributes
        self.target = target
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size


class Dataset(metaclass=abc.ABCMeta):
    """
    Abstract base class for a Dataset.

    Attributes
    ----------
    features_mapping : dict
        A dictionary mapping the original feature names to the encoded feature names.
    targets_mapping : dict
        A dictionary mapping the original target names to the encoded target names.
    protected_attributes : pd.DataFrame
        A DataFrame containing the protected attributes.
    sampled_indexes : list
        A list of indexes that have been sampled.
    error_flag : bool
        A flag indicating if an error has occurred.

    Methods
    -------
    __init__(self, config: DatasetConfig):
        Initializes the Dataset object with the provided dataset information.
    _load_dataset(self):
        Abstract method to load the dataset.
    _transform_dataset(self):
        Abstract method to transform the dataset. The specific transformations depend on the
        dataset and can include operations such as deriving new attributes, renaming attributes,
        or encoding attribute values.
    save_protected_attributes(self):
        Saves the protected attributes in the dataset.
    set_feature(self, feature: str, series: pd.DataFrame):
        Sets a feature in the dataset.
    get_protected_attributes(self):
        Gets the protected attributes in the dataset.
    get_protected_feature(self, feature) -> pd.DataFrame:
        Gets a protected feature in the dataset.
    get_dummy_protected_feature(self, feature) -> pd.DataFrame:
        Gets a dummy encoded protected feature in the dataset.
    get_value_counts(self, feature):
        Gets the value counts of a feature in the dataset.
    merge_features_and_targets(self) -> (pd.DataFrame, str):
        Merges the features and targets into a single DataFrame.
    split(self):
        Splits the dataset into train, validation, and test sets.
    print_dataset(self):
        Prints the dataset.
    _preprocessing(self):
        Preprocesses the dataset.
    __drop_invalid_instances(self):
        Drops invalid instances from the dataset.
    __parse_dataset_config(self, config: DatasetConfig):
        Parses the dataset configuration.
    __reset_indexes(self):
        Resets the indexes of the dataset.
    __quantize_numerical_features(self, n_bins=4):
        Quantizes the numerical features of the dataset.
    __label_encode_categorical(self, features=False, targets=True):
        Label encodes the categorical features and targets of the dataset.
    """

    def __init__(self, config: DatasetConfig):
        """
        Initializes the Dataset object with the provided dataset information.

        Parameters
        ----------
        config : DatasetConfig
            The configuration information for the dataset.
        """
        self.features_mapping = None
        self.targets_mapping = None
        self.protected_attributes = None
        self.sampled_indexes = None
        self.error_flag = False

        self.__parse_dataset_config(config)

        # load raw data into dataframe
        self.features, self.targets = self._load_dataset()

        # preprocess dataset
        self._preprocessing()

        self.save_protected_attributes()

        logger.info(f'[{extract_filename(__file__)}] Loaded.')

    @abc.abstractmethod
    def _load_dataset(self):
        """
        Abstract method to load the dataset.
        """

    @abc.abstractmethod
    def _transform_dataset(self):
        """
        Abstract method to transform the dataset. The specific transformations depend on the dataset and can include
        operations such as deriving new attributes, renaming attributes, or encoding attribute values.
        """

    def save_protected_attributes(self):
        """
        Saves the protected attributes in the dataset. The protected attributes are stored in a separate DataFrame
        for easy access and manipulation.
        """
        self.protected_attributes = self.features.loc[:, self.protected_features_names]

    def set_feature(self, feature: str, series: pd.DataFrame):
        """
        Sets a feature in the dataset. This method allows you to modify a specific feature in the dataset.

        Parameters
        ----------
        feature : str
            The name of the feature to be set.
        series : pd.DataFrame
            The new values for the feature.
        """
        if feature not in self.features.columns:
            raise ValueError(f'Feature {feature} does not exist.')

        self.features[feature] = series

    def get_protected_attributes(self) -> pd.DataFrame:
        """
        Gets the protected attributes in the dataset.

        Returns
        -------
        pd.DataFrame
            The protected attributes.
        """
        return self.protected_attributes.loc[:, self.protected_features_names]

    def get_protected_feature(self, feature) -> pd.DataFrame:
        """
        Gets a protected feature in the dataset. The protected feature is returned as a DataFrame.

        Parameters
        ----------
        feature : str
            The name of the protected feature to be retrieved.

        Returns
        -------
        pd.DataFrame
            The protected feature.
        """

        if feature not in self.protected_features_names:
            raise ValueError(f'Feature {feature} is not a protected feature.')

        return self.protected_attributes[feature]

    def get_dummy_protected_feature(self, feature) -> pd.DataFrame:
        """
        Gets a dummy encoded protected feature in the dataset. The protected feature is returned as a DataFrame
        with dummy encoding.

        Parameters
        ----------
        feature : str
            The name of the protected feature to be retrieved.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the dummy encoded protected feature.
        """
        if feature not in self.protected_features_names:
            raise ValueError(f'Attribute {feature} is not a protected attribute.')

        protected_feature = self.get_protected_feature(feature)
        dummy_protected_feature = pd.get_dummies(protected_feature, dtype=float)
        dummy_protected_feature = dummy_protected_feature.rename(columns=self.features_mapping[feature])

        return dummy_protected_feature

    def merge_features_and_targets(self) -> (pd.DataFrame, str):
        """
        Merges the features and targets into a single DataFrame. The merged DataFrame and the name of the target
        column are returned.

        Returns
        -------
        pd.DataFrame, str
            The DataFrame containing the merged features and targets, and the name of the target column.
        """
        data = pd.concat([self.features, self.targets], axis='columns')
        outcome = self.targets.columns[0]

        return data, outcome

    def split(self):
        """
        Splits the dataset into train, validation, and test sets. The split datasets are returned as separate Dataset
        objects. The split is stratified based on the protected attributes and the target.
        """

        # split into train and validation/test sets
        stratify_criteria = pd.concat([self.features[self.protected_features_names], self.targets], axis=1)

        # Assuming 'subgroup_column' contains the subgroup identifiers
        subgroup_counts = stratify_criteria.value_counts()

        # Identify subgroups with dimensionality 1
        subgroups_to_remove = subgroup_counts[subgroup_counts == 1].index

        indexes_to_remove = []
        columns = subgroups_to_remove.names
        for i in range(len(subgroups_to_remove)):
            df = copy.deepcopy(stratify_criteria)
            for j in range(len(columns)):
                df = df[df[columns[j]] == subgroups_to_remove[i][j]]

            indexes_to_remove.append(df.index.to_list()[0])

        self.features = self.features.drop(index=indexes_to_remove)
        self.targets = self.targets.drop(index=indexes_to_remove)

        stratify_criteria = stratify_criteria.drop(index=indexes_to_remove)

        x_train, x_test, y_train, y_test = train_test_split(self.features, self.targets,
                                                            train_size=self.train_size,
                                                            random_state=get_seed(),
                                                            shuffle=True,
                                                            stratify=stratify_criteria)

        # split into validation and test sets
        split_ratio = self.test_size / (self.validation_size + self.test_size)
        stratify_criteria = pd.concat([x_test[self.protected_features_names], y_test], axis=1)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                        test_size=split_ratio,
                                                        random_state=get_seed(),
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

    def _preprocessing(self):
        """
        Preprocesses the dataset. This method performs several preprocessing steps on the dataset, including dropping
        invalid instances, quantizing numerical features, transforming the dataset, and label encoding categorical
        features and targets.
        """
        logger.info("[DATASET] Pre-processing features and targets.")

        self.__drop_invalid_instances()
        self.__quantize_numerical_features()
        self._transform_dataset()
        self.__label_encode_categorical(features=True, targets=True)

        logger.info("[DATASET] Pre-processing complete.")

    def __drop_invalid_instances(self):
        """
        Drops invalid instances from the dataset. An instance is considered invalid if it contains any missing values.
        """
        drop_indexes = [index for index, row in self.features.iterrows() if row.isnull().any()]
        self.features = self.features.drop(drop_indexes).reset_index(drop=True)
        self.targets = self.targets.drop(drop_indexes).reset_index(drop=True)

    def __parse_dataset_config(self, config: DatasetConfig):
        """
        Parses the dataset configuration. This method extracts the necessary information from the provided DatasetConfig
        object and stores it in the Dataset object.

        Parameters
        ----------
        config : DatasetConfig
            The configuration information for the dataset.
        """

        self.name = config.name
        self.protected_features_names = config.protected_attributes
        self.target = config.target
        self.train_size = config.train_size
        self.validation_size = config.validation_size
        self.test_size = config.test_size

    def __reset_indexes(self):
        """
        Resets the indexes of the dataset. This method ensures that the indexes of the features and targets DataFrames
        are in sync after any operations that may have modified them.
        """
        self.features = self.features.reset_index(drop=True)
        self.targets = self.targets.reset_index(drop=True)

    def __quantize_numerical_features(self, n_bins=4):
        """
        Quantizes the numerical features of the dataset. This method divides the range of each numerical feature into
        equal width bins and replaces the original values with the bin numbers.

        Parameters
        ----------
        n_bins : int, optional
            The number of bins to divide each numerical feature into. The default is 4.
        """
        numerical_data = self.features.select_dtypes(include=['number'])
        to_drop = []
        for attribute in self.protected_features_names:
            if attribute in numerical_data.columns:
                to_drop.append(attribute)
        numerical_data = numerical_data.drop(columns=to_drop)
        categorical_data = self.features.drop(numerical_data, axis=1)

        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', random_state=get_seed())
        discretized_numerical_data = pd.DataFrame(discretizer.fit_transform(numerical_data),
                                                  columns=numerical_data.columns)
        processed_features = pd.concat([categorical_data, discretized_numerical_data], axis=1)

        self.features = processed_features

    def __label_encode_categorical(self, features=False, targets=True):
        """
        Label encodes the categorical features and targets of the dataset. This method replaces the original categorical
        values with numerical labels.

        Parameters
        ----------
        features : bool, optional
            Whether to label encode the features. The default is False.
        targets : bool, optional
            Whether to label encode the targets. The default is True.
        """

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


def update_dataset(dataset: Dataset, features: np.ndarray = None, targets: np.ndarray = None) -> Dataset:
    """
    Updates the features and targets of a given dataset.

    Parameters
    ----------
    dataset : Dataset
        The original dataset to be updated.
    features : np.ndarray, optional
        The new features to be set in the dataset. If not provided, the original features are kept.
    targets : np.ndarray, optional
        The new targets to be set in the dataset. If not provided, the original targets are kept.

    Returns
    -------
    updated_dataset : Dataset
        The updated dataset with the new features and targets.
    """
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


def match_features(dataset_a: Dataset, dataset_b: Dataset) -> Dataset:
    """
    Matches the features of two datasets.

    Parameters
    ----------
    dataset_a : Dataset
        The first dataset.
    dataset_b : Dataset
        The second dataset.

    Returns
    -------
    df : Dataset
        The second dataset with its features updated to match the features of the first dataset.
    """
    df = copy.deepcopy(dataset_b)
    common_columns = dataset_a.features.columns.intersection(dataset_b.features.columns)
    df.features = dataset_b.features.drop(columns=dataset_b.features.columns.difference(common_columns))

    return df

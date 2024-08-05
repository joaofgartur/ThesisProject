import abc
import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from utils import DatasetConfig
from utils.logging import logger
from utils.files import extract_filename
from utils.random_numbers import get_seed


class Dataset(metaclass=abc.ABCMeta):
    """
    Abstract base class for a Dataset.

    Attributes
    ----------
    features_mapping : dict
        A dictionary mapping the original feature names to the encoded feature names.
    targets_mapping : dict
        A dictionary mapping the original target names to the encoded target names.
    protected_features : pd.DataFrame
        A DataFrame containing the protected attributes.
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
    save_protected_features(self):
        Saves the protected attributes in the dataset.
    set_feature(self, feature: str, series: pd.DataFrame):
        Sets a feature in the dataset.
    get_protected_features(self):
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
        self.protected_features = None
        self.error_flag = False

        self.__parse_dataset_config(config)
        self.features, self.targets = self._load_dataset()
        self._preprocessing()
        self.save_protected_features()

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

    def update(self, features: np.ndarray = None, targets: np.ndarray = None):

        if features is not None:
            self.features = pd.DataFrame(features, columns=self.features.columns)
            self.features.reset_index(drop=True, inplace=True)

        if targets is not None:
            self.targets = pd.DataFrame(targets, columns=self.targets.columns)
            self.targets.reset_index(drop=True, inplace=True)

    def save_protected_features(self):
        """
        Saves the protected attributes in the dataset. The protected attributes are stored in a separate DataFrame
        for easy access and manipulation.
        """
        self.protected_features = self.features.loc[:, self.protected_features_names]

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
        self.features[feature] = series

    def get_protected_features(self) -> pd.DataFrame:
        """
        Gets the protected attributes in the dataset.

        Returns
        -------
        pd.DataFrame
            The protected attributes.
        """
        return self.protected_features.loc[:, self.protected_features_names]

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
        try:
            return self.protected_features[feature]
        except KeyError:
            raise ValueError(f'Feature {feature} is not a protected feature.')

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
        try:
            protected_feature = self.get_protected_feature(feature)
            dummy_protected_feature = pd.get_dummies(protected_feature, dtype=float)
            return dummy_protected_feature.rename(columns=self.features_mapping[feature])
        except (KeyError, ValueError):
            raise ValueError(f'Feature {feature} is not a protected feature.')

    def split(self, sensitive_attributes: list[str] = None):
        """
        Splits the dataset into train, validation, and test sets. The split datasets are returned as separate Dataset
        objects. The split is stratified based on the protected attributes and the target.
        """

        # split into train and validation/test sets
        if sensitive_attributes is None:
            sensitive_attributes = self.protected_features_names

        stratify_criteria = pd.concat([self.features[sensitive_attributes], self.targets], axis=1)
        x_train, x_test, y_train, y_test = train_test_split(self.features, self.targets,
                                                            train_size=self.train_size,
                                                            random_state=get_seed(),
                                                            shuffle=True,
                                                            stratify=stratify_criteria)

        split_ratio = self.test_size / (self.validation_size + self.test_size)
        stratify_criteria = pd.concat([x_test[sensitive_attributes], y_test], axis=1)

        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                        test_size=split_ratio,
                                                        random_state=get_seed(),
                                                        shuffle=True,
                                                        stratify=stratify_criteria)

        train_set = copy.deepcopy(self)
        train_set.update(features=x_train, targets=y_train)
        train_set.save_protected_features()

        validation_set = copy.deepcopy(self)
        validation_set.update(features=x_val, targets=y_val)
        validation_set.save_protected_features()

        test_set = copy.deepcopy(self)
        test_set.update(features=x_test, targets=y_test)
        test_set.save_protected_features()

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
        invalid_indexes = self.features[self.features.isnull().any(axis=1)].index
        self.features.drop(invalid_indexes, inplace=True)
        self.features.reset_index(drop=True, inplace=True)
        self.targets.drop(invalid_indexes, inplace=True)
        self.targets.reset_index(drop=True, inplace=True)

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
        self.protected_features_names = config.protected_features
        self.target = config.target
        self.train_size = config.train_size
        self.validation_size = config.validation_size
        self.test_size = config.test_size

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
        numerical_data = numerical_data.drop(columns=self.protected_features_names, errors='ignore')
        categorical_data = self.features.drop(numerical_data, axis=1)

        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', random_state=get_seed())
        numerical_data = pd.DataFrame(discretizer.fit_transform(numerical_data), columns=numerical_data.columns)

        self.features = pd.concat([categorical_data, numerical_data], axis=1)

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

        def label_encode(df: pd.DataFrame):
            mapping = {}
            encoded_df = pd.DataFrame(df, columns=df.columns)

            columns_to_encode = encoded_df.select_dtypes(include=['object']).columns
            for column in columns_to_encode:
                unique_vals, encoded_labels = np.unique(encoded_df[column], return_inverse=True)
                encoded_df[column] = encoded_labels
                mapping[column] = {i: val for i, val in enumerate(unique_vals)}

            return encoded_df, mapping

        if features:
            self.features, self.features_mapping = label_encode(self.features)

        if targets:
            self.targets, self.targets_mapping = label_encode(self.targets)

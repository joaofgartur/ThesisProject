import abc
from datasets import convert_categorical_into_numerical, remove_invalid_columns


class Dataset(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, dataset_name, sensitive_attributes_labels=None):
        """Initializes class instance"""
        if sensitive_attributes_labels is None:
            sensitive_attributes_labels = []
        self.dataset_name = dataset_name
        self.sensitive_attributes = sensitive_attributes_labels
        self.features = None
        self.targets = None
        self.features_mapping = None
        self.target_mapping = None

    @abc.abstractmethod
    def get_sensitive_attributes(self):
        """"""

    @abc.abstractmethod
    def _load_dataset(self):
        """Loads the dataset"""

    @abc.abstractmethod
    def _transform_dataset(self):
        """"""

    def _prepare_dataset(self):
        # drop any instances with empty values
        self.features, removed_indexes = remove_invalid_columns(self.features, [])
        self.targets, _ = remove_invalid_columns(self.targets, removed_indexes)

        # transform attributes
        self._transform_dataset()

        # convert categorical into numerical
        self.features, self.features_mapping = convert_categorical_into_numerical(self.features)
        self.targets, self.target_mapping = convert_categorical_into_numerical(self.targets)

    def print_dataset(self):
        """Prints the dataset"""
        print(self.features)
        print(self.targets)

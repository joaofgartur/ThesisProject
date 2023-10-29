import abc
from sklearn.preprocessing import LabelEncoder


def convert_categorical_into_numerical(dataframe):
    label_encoder = LabelEncoder()
    labels_mapping = {}

    for column in dataframe.columns:
        if dataframe[column].dtype == object:
            dataframe[column] = label_encoder.fit_transform(dataframe[column])
            mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            labels_mapping.update({column: mapping})

    return dataframe, labels_mapping


def remove_invalid_columns(dataframe, indexes=None):
    if indexes is None:
        indexes = []
    if len(indexes) == 0:
        indexes = [index for index, row in dataframe.iterrows() if row.isnull().any()]
    dataframe = dataframe.drop(indexes)
    return dataframe, indexes


class Dataset(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, dataset_name):
        """Initializes class instance"""
        self.dataset_name = dataset_name
        self.dataset = None
        self.features_mapping = None
        self.target_mapping = None

    @abc.abstractmethod
    def _load_dataset(self):
        """Loads the dataset"""

    @abc.abstractmethod
    def _prepare_dataset(self):
        """Removes invalid instances from the dataset."""

    @abc.abstractmethod
    def print_dataset(self):
        """Prints metadata"""

    @abc.abstractmethod
    def print_metadata(self):
        """"""

    @abc.abstractmethod
    def get_features(self):
        """"""

    @abc.abstractmethod
    def get_features_mapping(self):
        """"""

    @abc.abstractmethod
    def get_target(self):
        """"""

    @abc.abstractmethod
    def get_target_mapping(self):
       """"""

    @abc.abstractmethod
    def get_sensitive_attributes(self):
        """"""

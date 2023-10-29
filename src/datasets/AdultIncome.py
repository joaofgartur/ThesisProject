from constants import NEGATIVE_OUTCOME, POSITIVE_OUTCOME
from datasets import Dataset, remove_invalid_columns, convert_categorical_into_numerical
from ucimlrepo import fetch_ucirepo


class AdultIncome(Dataset):

    def __init__(self):
        self._load_dataset()

    def _load_dataset(self):
        self.dataset = fetch_ucirepo(id=2)
        self._prepare_dataset()

    def _prepare_dataset(self):
        # drop any instances with empty values
        self.dataset.data.features, removed_indexes = remove_invalid_columns(self.dataset.data.features,
                                                                             [])
        self.dataset.data.targets, _ = remove_invalid_columns(self.dataset.data.targets,
                                                              removed_indexes)

        self.__derive_classes__()

        # convert categorical into numerical
        self.dataset.data.features, self.features_mapping = convert_categorical_into_numerical(
            self.dataset.data.features)
        self.dataset.data.targets, self.target_mapping = convert_categorical_into_numerical(self.dataset.data.targets)

    def __derive_classes__(self):
        self.dataset.data.targets = (self.dataset.data.targets
                                     .replace("<=", NEGATIVE_OUTCOME, regex=True)
                                     .replace(">", POSITIVE_OUTCOME, regex=True)
                                     .astype('int64'))

    def get_features(self):
        return self.dataset.data.features

    def get_features_mapping(self):
        return self.features_mapping

    def get_target(self):
        return self.dataset.data.targets

    def get_target_mapping(self):
        return self.target_mapping

    def get_sensitive_attributes(self):
        return self.dataset.data.features.loc[:, ["race", "sex"]]

    def print_metadata(self):
        print(self.dataset.metadata)

    def print_dataset(self):
        print(self.dataset.data.features)

from datasets import Dataset, remove_invalid_columns, convert_categorical_into_numerical
from ucimlrepo import fetch_ucirepo
from constants import PRIVILEGED, UNPRIVILEGED


class GermanCredit(Dataset):

    def __init__(self):
        self._load_dataset()

    def _load_dataset(self):
        self.dataset = fetch_ucirepo(id=144)
        self._prepare_dataset()

    def _prepare_dataset(self):
        # drop any instances with empty values
        self.dataset.data.features, removed_indexes = remove_invalid_columns(self.dataset.data.features,
                                                                             [])
        self.dataset.data.targets, _ = remove_invalid_columns(self.dataset.data.targets,
                                                              removed_indexes)

        # transform attributes
        self._derive_sex()

        # convert categorical into numerical
        self.dataset.data.features, self.features_mapping = convert_categorical_into_numerical(self.dataset.data.features)
        self.dataset.data.targets, self.target_mapping = convert_categorical_into_numerical(self.dataset.data.targets)

    def _derive_sex(self):
        # replace attribute 9 values by either "male" or "female"
        male_values = ["A91", "A93", "A94"]
        female_values = ["A92", "A95"]

        for v in male_values:
            self.dataset.data.features['Attribute9'] = self.dataset.data.features['Attribute9'].replace(v, PRIVILEGED)

        for v in female_values:
            self.dataset.data.features['Attribute9'] = self.dataset.data.features['Attribute9'].replace(v, UNPRIVILEGED)

    def get_features(self):
        return self.dataset.data.features

    def get_features_mapping(self):
        return self.features_mapping

    def get_target(self):
        return self.dataset.data.targets

    def get_target_mapping(self):
        return self.target_mapping

    def get_sensitive_attributes(self):
        return self.dataset.data.features.loc[:, ["Attribute9", "Attribute13"]]

    def print_metadata(self):
        print(self.dataset.metadata)

    def print_dataset(self):
        print(self.dataset.data.features)

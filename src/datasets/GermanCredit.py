from datasets import Dataset, remove_invalid_columns, convert_categorical_into_numerical
from ucimlrepo import fetch_ucirepo
from constants import PRIVILEGED, UNPRIVILEGED, NEGATIVE_OUTCOME


class GermanCredit(Dataset):

    def __init__(self, dataset_name, sensitive_attributes_labels):
        Dataset.__init__(self, dataset_name, sensitive_attributes_labels)
        self._load_dataset()
        self._prepare_dataset()

    def _load_dataset(self):
        dataset = fetch_ucirepo(id=144)
        self.features = dataset.data.features
        self.targets = dataset.data.targets

    def _transform_dataset(self):
        self.__derive_sex__()
        self.__derive_classes__()

    def __derive_sex__(self):
        # replace attribute 9 values by either "male" or "female"
        male_values = ["A91", "A93", "A94"]
        female_values = ["A92", "A95"]

        for v in male_values:
            self.features['Attribute9'] = self.features['Attribute9'].replace(v, PRIVILEGED)

        for v in female_values:
            self.features['Attribute9'] = self.features['Attribute9'].replace(v, UNPRIVILEGED)

    def __derive_classes__(self):
        # set the unfavoured classification to zero
        self.targets = self.targets.rename(columns={"class": "label"})
        self.targets = self.targets.replace(2, NEGATIVE_OUTCOME)

    def get_sensitive_attributes(self):
        return self.features.loc[:, ["Attribute9"]]

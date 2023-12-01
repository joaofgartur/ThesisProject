import pandas as pd
from ucimlrepo import fetch_ucirepo

from datasets import Dataset
from constants import NEGATIVE_OUTCOME, PRIVILEGED, UNPRIVILEGED, POSITIVE_OUTCOME


class GermanCredit(Dataset):
    _LOCAL_DATA_FILE = "datasets/local_storage/german_credit/german.data"

    def __init__(self, dataset_info: dict):
        Dataset.__init__(self, dataset_info)
        self._load_dataset()
        self._prepare_dataset()

    def _load_dataset(self):
        try:
            dataset = fetch_ucirepo(id=144)
            self.features = dataset.data.features
            self.targets = dataset.data.targets
        except ConnectionError:
            dataset = pd.read_csv(self._LOCAL_DATA_FILE, header=None, sep=" ")
            labels = ['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5',
                      'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9', 'Attribute10',
                      'Attribute11', 'Attribute12', 'Attribute13', 'Attribute14', 'Attribute15',
                      'Attribute16', 'Attribute17', 'Attribute18', 'Attribute19', 'Attribute20',
                      'class']
            dataset = dataset.set_axis(labels=labels, axis="columns")

            self.targets = pd.DataFrame(dataset["class"])
            self.features = dataset.drop(columns=["class"])

    def _transform_dataset(self):

        def binarize_attribute(x, y):
            if x == y:
                return PRIVILEGED
            return UNPRIVILEGED

        def derive_sex(x):
            male_values = ["A91", "A93", "A94"]
            if x in male_values:
                return "Male"
            return "Female"

        def derive_class(x):
            if x == 2:
                return NEGATIVE_OUTCOME
            return POSITIVE_OUTCOME

        self.features["Attribute9"] = self.features["Attribute9"].apply(lambda x: derive_sex(x))
        for attribute, value in self.sensitive_attributes_info.items():
            self.features[attribute] = self.features[attribute].apply(lambda x, y=value: binarize_attribute(x, y))

        self.targets = self.targets.rename(columns={"class": "label"})
        self.targets["label"] = self.targets["label"].apply(lambda x: derive_class(x))

import pandas as pd
from ucimlrepo import fetch_ucirepo

from datasets import Dataset
from constants import NEGATIVE_OUTCOME


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
        self.__define_privileged_and_unprivileged__()
        self.__derive_sex__()
        self.__derive_classes__()

    def __derive_sex__(self):
        _MALE = "Male"
        _FEMALE = "Female"
        _SEX_LABEL = "Attribute9"

        male_values = ["A91", "A93", "A94"]
        female_values = ["A92", "A95"]

        for v in male_values:
            self.features[_SEX_LABEL] = self.features[_SEX_LABEL].replace(v, _MALE)

        for v in female_values:
            self.features[_SEX_LABEL] = self.features[_SEX_LABEL].replace(v, _FEMALE)

    def __derive_classes__(self):
        # set the unfavoured classification to zero
        self.targets = self.targets.rename(columns={"class": "label"})
        self.targets = self.targets.replace(2, NEGATIVE_OUTCOME)

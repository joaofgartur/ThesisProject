from constants import NEGATIVE_OUTCOME, POSITIVE_OUTCOME
from datasets import Dataset
from ucimlrepo import fetch_ucirepo


class AdultIncome(Dataset):

    def __init__(self, dataset_name, sensitive_attributes_labels):
        Dataset.__init__(self, dataset_name, sensitive_attributes_labels)
        self._load_dataset()
        self._prepare_dataset()

    def _load_dataset(self):
        dataset = fetch_ucirepo(id=2)
        self.features = dataset.data.features
        self.targets = dataset.data.targets

    def _transform_dataset(self):
        self.__derive_classes__()

    def __derive_classes__(self):
        self.targets = (self.targets
                        .replace("<=", NEGATIVE_OUTCOME, regex=True)
                        .replace(">", POSITIVE_OUTCOME, regex=True)
                        .astype('int64'))

    def get_sensitive_attributes(self):
        return self.features.loc[:, self.sensitive_attributes]

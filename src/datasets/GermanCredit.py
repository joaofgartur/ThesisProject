import pandas as pd
from ucimlrepo import fetch_ucirepo

from constants import NEGATIVE_OUTCOME, POSITIVE_OUTCOME
from datasets import Dataset, DatasetConfig
from helpers import logger, extract_filename


class GermanCredit(Dataset):

    _LOCAL_DATA_FILE = "datasets/local_storage/german_credit/german.data"

    def __init__(self, config: DatasetConfig):
        logger.info(f'[{extract_filename(__file__)}] Loading...')
        Dataset.__init__(self, config)

    def _load_dataset(self):
        try:
            dataset = fetch_ucirepo(id=144)
            return dataset.data.features, dataset.data.targets
        except ConnectionError:
            logger.error(f'[{extract_filename(__file__)}] Failed to load from online source.'
                         f' Loading from local storage.')

            dataset = pd.read_csv(self._LOCAL_DATA_FILE, header=None, sep=" ")
            column_labels = ['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5',
                             'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9', 'Attribute10',
                             'Attribute11', 'Attribute12', 'Attribute13', 'Attribute14', 'Attribute15',
                             'Attribute16', 'Attribute17', 'Attribute18', 'Attribute19', 'Attribute20',
                             'class']
            dataset = dataset.set_axis(labels=column_labels, axis="columns")

            targets = pd.DataFrame(dataset[self.target])
            features = dataset.drop(columns=[self.target])

            return features, targets

    def _transform_protected_attributes(self):

        def derive_age(x, cutoff=25):
            return 'Young' if x < cutoff else 'Aged'

        def derive_sex(x):
            return 'Male' if x in ['A91', 'A93', 'A94'] else 'Female'

        def derive_class(x):
            if x == 2:
                return NEGATIVE_OUTCOME * 1.0
            return POSITIVE_OUTCOME * 1.0

        # derive sex values
        sex_column = 'Attribute9'
        age_column = 'Attribute13'

        self.features[sex_column] = self.features[sex_column].apply(lambda x: derive_sex(x))
        self.features[age_column] = self.features[age_column].apply(lambda x: derive_age(x))

        # rename target column
        self.targets = self.targets.rename(columns={self.target: 'target'})
        self.target = 'target'

        self.targets[self.target] = self.targets[self.target].apply(lambda x: derive_class(x))

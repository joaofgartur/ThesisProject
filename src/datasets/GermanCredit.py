import pandas as pd
from ucimlrepo import fetch_ucirepo

from constants import NEGATIVE_OUTCOME, POSITIVE_OUTCOME
from datasets import Dataset
from utils import logger, extract_filename, DatasetConfig


class GermanCredit(Dataset):
    """
    Representation of the German Credit dataset [1].

    Attributes
    ----------
    _LOCAL_DATA_FILE : str
        The local path to the dataset file.

    Methods
    -------
    __init__(self, config: DatasetConfig):
        Initializes the GermanCredit object with the provided dataset information.
    _load_dataset(self):
        Loads the dataset from an online source or local file, filters the data, and separates features and targets.
    _transform_protected_attributes(self):
        Transforms the protected attributes in the dataset.

    References
    ----------
    [1] Hans Hofmann. Statlog (German Credit Data). UCI Machine Learning Repository, 1994.
     DOI: https://doi.org/10.24432/C5NC77 URL https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
    """

    _LOCAL_DATA_FILE = "datasets/local_storage/german_credit/german.data"

    def __init__(self, config: DatasetConfig):
        """
        Initializes the GermanCredit object with the provided dataset information.

        Parameters
        ----------
        config : DatasetConfig
            The configuration information for the dataset.
        """
        logger.info(f'[{extract_filename(__file__)}] Loading...')
        Dataset.__init__(self, config)

    def _load_dataset(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Loads the dataset from an online source or local file, filters the data, and separates features and targets.

        Returns
        -------
        features : pd.DataFrame
            The features from the dataset.
        targets : pd.DataFrame
            The target values from the dataset.
        """
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

    def _transform_dataset(self):
        """
        Transforms the dataset.

        The transformations include deriving age and sex values, and renaming the target column. The sensitive
        attributes were selected following the study of the dataset in [2]. The sex feature was transformed according to
        the dataset's documentation [1]. The age feature was transformed based on the age cutoff
        provided in [2].

        References
        ----------
        [2] Tai Le Quy, Arjun Roy, Vasileios Iosifidis, and Eirini Ntoutsi. A survey on datasets for fairness-aware
             machine learning. CoRR, abs/2110.00530, 2021. URL https://arxiv.org/abs/2110.00530.
        """

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

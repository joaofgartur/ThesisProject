import pandas as pd

from datasets import Dataset
from utils import extract_filename, logger, DatasetConfig


class LawSchoolAdmissions(Dataset):
    """
    Representation of the Law School dataset [1].

    Attributes
    ----------
    _LOCAL_DATA_FILE : str
        The local path to the dataset file.

    Methods
    -------
    __init__(self, config: DatasetConfig):
        Initializes the LawSchoolAdmissions object with the provided dataset information.
    _load_dataset(self):
        Loads the dataset from an online source or local file, filters the data, and separates features and targets.
    _transform_protected_attributes(self):
        Transforms the protected attributes in the dataset. Currently, this method does not perform any transformations.

    References
    ----------
    [1] Wightman, L. F. (1998). LSAC national longitudinal bar passage study. LSAC research report series.
    """

    _LOCAL_DATA_FILE = "datasets/local_storage/law_school_admissions/bar_pass_prediction.csv"

    def __init__(self, config: DatasetConfig):
        """
        Initializes the LawSchoolAdmissions object with the provided dataset information.

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
            dataset_url = 'https://storage.googleapis.com/lawschool_dataset/bar_pass_prediction.csv'
            data = pd.read_csv(dataset_url)
        except (FileNotFoundError, ValueError):
            logger.error(f'[{extract_filename(__file__)}] Failed to load from online source.'
                         f' Loading from local storage.')
            data = pd.read_csv(self._LOCAL_DATA_FILE)

        def filter_data(_data: pd.DataFrame) -> pd.DataFrame:
            """
            Filters the data to keep only the necessary features according to the features selected in [2].
            The columns of sensitive features for this dataset were chosen from the exploratory data analysis in [3].

            Parameters
            ----------
            _data : pd.DataFrame
                The original dataset.

            Returns
            -------
            _data : pd.DataFrame
                The filtered dataset.

            References
            ----------
            [2] Tai Le Quy, Arjun Roy, Vasileios Iosifidis, and Eirini Ntoutsi. A survey on datasets for fairness-aware
             machine learning. CoRR, abs/2110.00530, 2021. URL https://arxiv.org/abs/2110.00530.
            [3] eds8531. LSAC Dataset EDA and Predictions. Kaggle,
             URL https://www.kaggle.com/code/eds8531/lsac-dataset-eda-and-predictions/notebook.
            """
            features_to_keep = ['decile1b', 'decile3', 'lsat', 'ugpa', 'zgpa', 'zfygpa', 'fulltime', 'fam_inc',
                                'gender', 'tier', 'race1', 'pass_bar']
            return _data[features_to_keep]

        data = filter_data(data)

        targets = pd.DataFrame(data[self.target])
        features = data.drop(columns=[self.target])

        return features, targets

    def _transform_dataset(self):
        """
        Transforms the dataset. Currently, this method does not perform any transformations.
        """

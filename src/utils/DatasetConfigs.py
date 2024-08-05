class DatasetConfig:
    """
    Configuration class for a Dataset.

    Attributes
    ----------
    name : str
        The name of the dataset.
    protected_features : list[str]
        The list of protected attributes in the dataset.
    target : str
        The target attribute in the dataset.
    train_size : float
        The proportion of the dataset to include in the train split.
    validation_size : float
        The proportion of the dataset to include in the validation split.
    test_size : float
        The proportion of the dataset to include in the test split.
    """

    def __init__(self, name: str, protected_features: list[str], target: str,
                 train_size: float, test_size: float, validation_size: float):
        """
        Initializes the DatasetConfig object with the provided dataset information.

        Parameters
        ----------
        name : str
            The name of the dataset.
        protected_features : list[str]
            The list of protected attributes in the dataset.
        target : str
            The target attribute in the dataset.
        train_size : float
            The proportion of the dataset to include in the train split.
        validation_size : float
            The proportion of the dataset to include in the validation split.
        test_size : float
            The proportion of the dataset to include in the test split.
        """
        self.name = name
        self.protected_features = protected_features
        self.target = target
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size

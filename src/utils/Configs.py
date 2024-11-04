"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""

from configparser import ConfigParser, NoSectionError, NoOptionError

from .DatasetConfigs import DatasetConfig
from .logging import logger
from .random_numbers import get_seed, set_seed
from .classifiers import enable_gpu_acceleration, set_gpu_device, set_gpu_allocated_memory, disable_gpu_acceleration, \
    set_surrogate_classifiers

global_configs = None


class Configs(object):
    """
    Class to manage the configuration settings for the bias correction project.

    Attributes
    ----------
    global_configs_file : str
        Path to the global configuration file.
    algorithms_configs_file : str
        Path to the algorithms configuration file.
    dataset_configs_file : str
        Path to the dataset configuration file.
    num_runs : int
        Number of runs for the algorithms.
    num_iterations : int
        Number of iterations for the algorithms.
    gpu_device_id : int
        ID of the GPU device to be used.

    Methods
    -------
    __init__(configs_file: str):
        Initializes the Configs object with the provided configuration file.
    __parse_global_configs_file():
        Parses the global configuration file.
    __config_surrogate_classifiers(parser: ConfigParser):
        Configures the surrogate classifiers.
    __config_gpu_acceleration(parser: ConfigParser):
        Configures GPU acceleration settings.
    set_global_configs_file(configurations_file: str):
        Sets the global configuration file.
    set_algorithms_configs_file(configurations_file: str):
        Sets the algorithms configuration file.
    set_dataset_configs_file(configurations_file: str):
        Sets the dataset configuration file.
    get_global_configs_file() -> str:
        Gets the global configuration file path.
    get_algorithms_configs_file() -> str:
        Gets the algorithms configuration file path.
    get_dataset_configs_file() -> str:
        Gets the dataset configuration file path.
    get_dataset_configs(dataset: str) -> DatasetConfig:
        Gets the dataset configuration for the specified dataset.
    """

    def __init__(self, configs_file: str):
        """
        Initializes the Configs object with the provided configuration file.

        Parameters
        ----------
        configs_file : str
            Path to the global configuration file.
        """

        self.global_configs_file = configs_file
        self.algorithms_configs_file = None
        self.dataset_configs_file = None

        self.num_runs = 0
        self.num_iterations = 0
        self.gpu_device_id = None

        self.__parse_global_configs_file()

    def __parse_global_configs_file(self):
        """
        Parses the global configuration file.
        """

        try:
            parser = ConfigParser()
            parser.read(self.global_configs_file)

            header = 'global'
            self.dataset_configs_file = parser.get(header, 'dataset_configs').strip('"')
            self.algorithms_configs_file = parser.get(header, 'algorithms_configs').strip('"')
            self.num_runs = parser.getint(header, 'num_runs')
            self.num_iterations = parser.getint(header, 'num_iterations_algorithms')

            if get_seed() is None:
                seed = parser.getint(header, 'seed')
                set_seed(seed)

            self.__config_surrogate_classifiers(parser)
            self.__config_gpu_acceleration(parser)

        except NoSectionError as e:
            logger.error(f'Section [{e.section}] does not exist in the configuration file.')
            raise ValueError
        except NoOptionError as e:
            logger.error(f'Option [{e.option}] does not exist in the configuration file.')
            raise ValueError

    def __config_surrogate_classifiers(self, parser: ConfigParser):
        """
        Configures the surrogate classifiers.

        Parameters
        ----------
        parser : ConfigParser
            The configuration parser.
        """
        header = 'surrogate_classifiers'
        surrogate_classifiers = []

        if parser.get(header, 'logistic_regression'):
            from sklearn.linear_model import LogisticRegression
            surrogate_classifiers.append(LogisticRegression(random_state=get_seed()))

        if parser.get(header, 'svm'):
            from sklearn.svm import SVC
            surrogate_classifiers.append(SVC(random_state=get_seed()))

        if parser.get(header, 'gaussian_nb'):
            from sklearn.naive_bayes import GaussianNB
            surrogate_classifiers.append(GaussianNB())

        if parser.get(header, 'decision_tree'):
            from sklearn.tree import DecisionTreeClassifier
            surrogate_classifiers.append(DecisionTreeClassifier(random_state=get_seed()))

        if parser.get(header, 'random_forest'):
            from sklearn.ensemble import RandomForestClassifier
            surrogate_classifiers.append(RandomForestClassifier(random_state=get_seed()))

        set_surrogate_classifiers(surrogate_classifiers)

    def __config_gpu_acceleration(self, parser: ConfigParser):
        """
        Configures GPU acceleration settings.

        Parameters
        ----------
        parser : ConfigParser
            The configuration parser.
        """

        header = 'gpu'
        enable_gpu = parser.getboolean(header, 'enable')

        if enable_gpu:
            enable_gpu_acceleration()
            device_id = parser.getint(header, 'device_id')
            set_gpu_device(device_id)
            self.gpu_device_id = device_id
            gpu_allocated_memory = parser.getint(header, 'allocated_memory')
            set_gpu_allocated_memory(gpu_allocated_memory)
        else:
            disable_gpu_acceleration()

    def set_global_configs_file(self, configurations_file: str):
        """
        Sets the global configuration file.

        Parameters
        ----------
        configurations_file : str
            Path to the global configuration file.
        """
        self.global_configs_file = configurations_file

    def set_algorithms_configs_file(self, configurations_file: str):
        """
        Sets the algorithms configuration file.

        Parameters
        ----------
        configurations_file : str
            Path to the algorithms configuration file.
        """
        self.algorithms_configs_file = configurations_file

    def set_dataset_configs_file(self, configurations_file: str):
        """
        Sets the dataset configuration file.

        Parameters
        ----------
        configurations_file : str
            Path to the dataset configuration file.
        """
        self.dataset_configs_file = configurations_file

    def get_global_configs_file(self):
        """
        Gets the global configuration file path.

        Returns
        -------
        str
            Path to the global configuration file.
        """
        return self.global_configs_file

    def get_algorithms_configs_file(self):
        """
        Gets the algorithms configuration file path.

        Returns
        -------
        str
            Path to the algorithms configuration file.
        """
        return self.algorithms_configs_file

    def get_dataset_configs_file(self):
        """
        Gets the dataset configuration file path.

        Returns
        -------
        str
            Path to the dataset configuration file.
        """
        return self.dataset_configs_file

    def get_dataset_configs(self, dataset: str):
        """
        Gets the dataset configuration for the specified dataset.

        Parameters
        ----------
        dataset : str
            The name of the dataset.

        Returns
        -------
        DatasetConfig
            The dataset configuration.
        """
        parser = ConfigParser()
        parser.read(self.dataset_configs_file)

        return DatasetConfig(name=parser.get(dataset, 'name'),
                             target=parser.get(dataset, 'target'),
                             protected_features=parser.get(dataset, 'protected_attributes').split(','),
                             train_size=parser.getfloat(dataset, 'train_size'),
                             validation_size=parser.getfloat(dataset, 'validation_size'),
                             test_size=parser.getfloat(dataset, 'test_size'))


def set_global_configs(configs: Configs):
    """
    Sets the global configuration object.

    Parameters
    ----------
    configs : Configs
        The global configuration object.
    """
    global global_configs
    global_configs = configs


def get_global_configs():
    """
    Gets the global configuration object.

    Returns
    -------
    Configs
        The global configuration object.
    """
    return global_configs

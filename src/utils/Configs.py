from configparser import ConfigParser, NoSectionError, NoOptionError

from .DatasetConfigs import DatasetConfig
from .logging import logger
from .random_numbers import get_seed, set_seed
from .classifiers import enable_gpu_acceleration, set_gpu_device, set_gpu_allocated_memory, disable_gpu_acceleration, \
    set_surrogate_classifiers

global_configs = None


class Configs(object):

    def __init__(self, configs_file: str):
        self.global_configs_file = configs_file
        self.algorithms_configs_file = None
        self.dataset_configs_file = None

        self.num_runs = 0
        self.num_iterations = 0

        self.__parse_global_configs_file()

    def __parse_global_configs_file(self):
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
        header = 'gpu'
        enable_gpu = parser.getboolean(header, 'enable')

        if enable_gpu:
            enable_gpu_acceleration()
            set_gpu_device(parser.getint(header, 'device_id'))
            gpu_allocated_memory = parser.getint(header, 'allocated_memory')
            set_gpu_allocated_memory(gpu_allocated_memory)
        else:
            disable_gpu_acceleration()

    def set_global_configs_file(self, configurations_file: str):
        self.global_configs_file = configurations_file

    def set_algorithms_configs_file(self, configurations_file: str):
        self.algorithms_configs_file = configurations_file

    def set_dataset_configs_file(self, configurations_file: str):
        self.dataset_configs_file = configurations_file

    def get_global_configs_file(self):
        return self.global_configs_file

    def get_algorithms_configs_file(self):
        return self.algorithms_configs_file

    def get_dataset_configs_file(self):
        return self.dataset_configs_file

    def get_dataset_configs(self, dataset: str):
        parser = ConfigParser()
        parser.read(self.dataset_configs_file)

        return DatasetConfig(name=parser.get(dataset, 'name'),
                             target=parser.get(dataset, 'target'),
                             protected_features=parser.get(dataset, 'protected_attributes').split(','),
                             train_size=parser.getfloat(dataset, 'train_size'),
                             validation_size=parser.getfloat(dataset, 'validation_size'),
                             test_size=parser.getfloat(dataset, 'test_size'))


def set_global_configs(configs: Configs):
    global global_configs
    global_configs = configs


def get_global_configs():
    return global_configs

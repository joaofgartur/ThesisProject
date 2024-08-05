import copy
import os

import pandas as pd

from algorithms.Algorithm import Algorithm
from constants import NUM_DECIMALS
from datasets import Dataset
from utils import logger, write_dataframe_to_csv, dict_to_dataframe, round_df, get_seed, ResourceManager
from .assessment import classifier_assessment


class Pipeline:

    def __init__(self, dataset: Dataset,
                 unbiasing_algorithms: [Algorithm],
                 surrogate_classifiers: list[object],
                 target_classifier: object,
                 num_iterations: int = 1) -> None:

        self.dataset = dataset

        self.train_set = None
        self.validation_set = None
        self.test_set = None

        self.unbiasing_algorithms = unbiasing_algorithms
        self.surrogate_classifiers = surrogate_classifiers
        self.target_classifier = target_classifier
        self.num_iterations = num_iterations

        self.metadata = {'algorithm': 'NA', 'set': 'NA', 'save_path': 'results', 'sensitive_attribute': 'NA'}

    def __pre_intervention__(self, train_data: Dataset, protected_attribute: str) -> (pd.DataFrame, pd.DataFrame):
        logger.info("[PRE-INTERVENTION] Assessment.")
        return self.__assess_sensitive_attribute(train_data, protected_attribute)

    def __post_intervention__(self, train_data: Dataset, protected_attribute: str) -> (pd.DataFrame, pd.DataFrame):
        logger.info("[POST-INTERVENTION] Assessment.")
        return self.__assess_sensitive_attribute(train_data, protected_attribute)

    def __handle_correction_error__(self, train_data: Dataset, protected_attribute: str) -> (
    pd.DataFrame, pd.DataFrame):
        logger.info("[ERROR] Correction error handling.")
        metrics, distribution = self.__assess_sensitive_attribute(train_data, protected_attribute)
        columns = [col for col in metrics.columns if col.startswith('fairness') or col.startswith('performance')]
        metrics[columns] = 0.0
        metrics['error'], distribution['error'] = True, True

        return metrics, distribution

    def __assess_sensitive_attribute(self, train: Dataset, sensitive_attribute: str) -> (pd.DataFrame, pd.DataFrame):

        def assess(to_predict: Dataset, test: bool = False) -> (pd.DataFrame, pd.DataFrame):
            return self.__assessment__(train, to_predict, sensitive_attribute, test)

        self.metadata['set'] = 'Validation'
        val_metrics, val_distro = assess(self.validation_set)
        val_final_metrics, val_final_distro = assess(self.validation_set, True)

        self.metadata['set'] = 'Test'
        test_metrics, test_distro = assess(self.test_set)
        test_final_metrics, test_final_distro = assess(self.test_set, True)

        metrics = pd.concat([val_metrics, test_metrics, val_final_metrics, test_final_metrics])
        distribution = pd.concat([val_distro, test_distro, val_final_distro, test_final_distro])

        return metrics, distribution

    def __assessment__(self,
                       train: Dataset,
                       validation: Dataset,
                       sensitive_attribute: str,
                       target_classifier: bool = False) -> (pd.DataFrame, pd.DataFrame):

        # Surrogate Classifiers
        metadata = {
            'dataset': self.dataset.name,
            'attribute': sensitive_attribute,
            'group': 'NA',
            'algorithm': self.metadata['algorithm'],
            'set': self.metadata['set'],
            'iterations': 0,
            'error': False,
            'classifier_type': 'Target' if target_classifier else 'Surrogate',
        }
        metadata_row = dict_to_dataframe(metadata)

        classifiers = [self.target_classifier] if target_classifier else self.surrogate_classifiers

        metrics, distributions = pd.DataFrame(), pd.DataFrame()
        for classifier in classifiers:
            x = copy.deepcopy(train)
            y = copy.deepcopy(validation)
            _, c_metrics, c_distro = classifier_assessment(classifier, x, y, sensitive_attribute)
            metrics = pd.concat([metrics, c_metrics]).reset_index(drop=True)
            distributions = pd.concat([distributions, c_distro]).reset_index(drop=True)

        metrics = pd.concat([pd.concat([metadata_row] * len(metrics), ignore_index=True), metrics],
                            axis=1).reset_index(drop=True)
        distributions = (pd.concat([pd.concat([metadata_row] * len(distributions), ignore_index=True),
                                    distributions], axis=1).reset_index(drop=True))

        return metrics, distributions

    def __set_sensitive_attribute(self, algorithm: Algorithm, data: Dataset, sensitive_attribute: str, key: str) \
            -> Dataset:
        if algorithm.is_binary:
            data.set_feature(sensitive_attribute, data.get_dummy_protected_feature(sensitive_attribute)[key])
        else:
            data.set_feature(sensitive_attribute, pd.DataFrame(data.get_protected_feature(sensitive_attribute),
                                                               columns=[sensitive_attribute])[key])
        return data

    def __set_protected_group_and_num_iterations__(self, metrics_df, distribution_df, protected_group, iteration):
        metrics_df['group'], distribution_df['group'] = ([protected_group] * metrics_df.shape[0],
                                                         [protected_group] * distribution_df.shape[0])
        metrics_df['iterations'], distribution_df['iterations'] = ([iteration] * metrics_df.shape[0],
                                                                   [iteration] * distribution_df.shape[0])
        return metrics_df, distribution_df

    def __bias_reduction__(self, train: Dataset, validation: Dataset,
                           algorithm: Algorithm, sensitive_attribute: str):

        logger.info(f'[INTERVENTION] Reducing bias for sensitive attribute {sensitive_attribute} with'
                    f' algorithm {algorithm.__class__.__name__}.')

        if algorithm.is_binary:
            protected_groups = train.get_dummy_protected_feature(sensitive_attribute)
        else:
            protected_groups = pd.DataFrame(train.get_protected_feature(sensitive_attribute),
                                            columns=[sensitive_attribute])

        for group in protected_groups:

            if algorithm.is_binary:
                logger.info(
                    f"[INTERVENTION] Correcting bias w.r.t. attribute {sensitive_attribute} for {group} with "
                    f"{algorithm.__class__.__name__}")

            transformed = copy.deepcopy(train)

            if algorithm.needs_auxiliary_data:
                algorithm.auxiliary_data = validation

            for i in range(self.num_iterations):
                logger.info(f"[INTERVENTION] Iteration {i + 1} / {self.num_iterations}.")
                transformed = self.__set_sensitive_attribute(algorithm, transformed,
                                                             sensitive_attribute, group)

                try:
                    algorithm.fit(transformed, sensitive_attribute)
                    transformed = algorithm.transform(transformed)
                except ValueError:
                    logger.error(
                        f'[INTERVENTION] Error occurred bias correction w.r.t. attribute {sensitive_attribute} '
                        f'for {group} with "{algorithm.__class__.__name__}"')
                    metrics, distribution = self.__handle_correction_error__(train, sensitive_attribute)
                else:
                    metrics, distribution = self.__post_intervention__(transformed, sensitive_attribute)

                metrics, distribution = self.__set_protected_group_and_num_iterations__(metrics, distribution, group,
                                                                                        i + 1)
                self.save_metrics_and_distribution(metrics, distribution)

            del transformed, metrics, distribution

    def save_metrics_and_distribution(self, metrics: pd.DataFrame, distribution: pd.DataFrame) -> None:
        self.save(metrics, os.path.join(self.metadata['save_path'], 'metrics'))
        distribution.fillna(0.0, inplace=True)
        self.save(distribution, os.path.join(self.metadata['save_path'], 'distributions'))

    def run(self) -> None:
        try:
            logger.info("[PIPELINE] Start.")
            resource_manager = ResourceManager()
            resource_manager.start()

            for sensitive_attribute in self.dataset.protected_features_names:

                self.metadata['sensitive_attribute'] = sensitive_attribute
                self.metadata['algorithm'] = 'NA'

                # split
                self.train_set, self.validation_set, self.test_set = self.dataset.split([sensitive_attribute])

                # pre-intervention
                metrics, distribution = self.__pre_intervention__(self.train_set, sensitive_attribute)
                self.save_metrics_and_distribution(metrics, distribution)
                del metrics, distribution

                # correction
                for unbiasing_algorithm in self.unbiasing_algorithms:
                    self.metadata['algorithm'] = unbiasing_algorithm.__class__.__name__
                    self.__bias_reduction__(self.train_set, self.validation_set, unbiasing_algorithm,
                                            sensitive_attribute)

            resource_manager.stop()
            logger.info("[PIPELINE] End.")

        except Exception as e:
            logger.error(f'An error occurred in the pipeline: \n {e}')
            raise

    def save(self, df: pd.DataFrame, path: str = 'results') -> None:
        filename = f'{self.dataset.name}_{get_seed()}_{self.metadata["sensitive_attribute"]}.csv'
        write_dataframe_to_csv(df=round_df(df, NUM_DECIMALS), filename=filename, path=path)

    def run_and_save(self, path: str = 'results') -> None:
        self.metadata['save_path'] = path
        self.run()

import copy
import itertools

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from algorithms.Algorithm import Algorithm
from constants import NUM_DECIMALS
from datasets import Dataset, update_dataset
from helpers import logger, write_dataframe_to_csv, dict_to_dataframe, round_df
from .assessment import classifier_assessment


class Pipeline:

    def __init__(self, dataset: Dataset,
                 unbiasing_algorithms: [Algorithm],
                 surrogate_classifiers: list[object],
                 test_classifier: object,
                 num_iterations: int = 1) -> None:

        self.dataset = dataset
        self.train_set = None
        self.validation_set = None
        self.test_set = None

        self.unbiasing_algorithms = unbiasing_algorithms

        self.surrogate_classifiers = surrogate_classifiers
        self.test_classifier = test_classifier

        self.metrics_dict = {}
        self.distribution_dict = {}

        self.num_iterations = num_iterations

        self.unbiasing_algorithm: str = 'NA'
        self.set: str = 'NA'

    def __scale__(self, scaler: MinMaxScaler):

        scaler.fit(self.train_set.features)

        self.train_data = update_dataset(self.train_set, features=scaler.transform(self.train_set.features))
        self.validation_data = update_dataset(self.validation_set,
                                              features=scaler.transform(self.validation_set.features))
        self.test_data = update_dataset(self.test_set, features=scaler.transform(self.test_set.features))

    def __pre_intervention__(self, train_data: Dataset) -> (pd.DataFrame, pd.DataFrame):
        logger.info("[PRE-INTERVENTION] Assessment.")

        all_metrics_df = pd.DataFrame()
        all_distribution_df = pd.DataFrame()
        for attribute in train_data.protected_attributes:
            metrics_df, distribution_df = self.__attribute_assessment(train_data, attribute)
            all_metrics_df = pd.concat([all_metrics_df, metrics_df])
            all_distribution_df = pd.concat([all_distribution_df, distribution_df])

        return all_metrics_df, all_distribution_df

    def __post_intervention__(self, train_data: Dataset, protected_attribute: str) -> (pd.DataFrame, pd.DataFrame):
        logger.info("[POST-INTERVENTION] Assessment.")
        return self.__attribute_assessment(train_data, protected_attribute)

    def __attribute_assessment(self, train_data: Dataset, protected_attribute: str) -> (pd.DataFrame, pd.DataFrame):
        self.set = 'Validation'
        val_surrogate_metrics, val_surrogate_distro = self.__assessment__(train_data, self.validation_set,
                                                                          protected_attribute)
        val_final_metrics, val_final_distro = self.__assessment__(train_data, self.validation_set, protected_attribute,
                                                                  True)

        self.set = 'Test'
        test_surrogate_metrics, test_surrogate_distro = self.__assessment__(train_data, self.test_set,
                                                                            protected_attribute)
        test_final_metrics, test_final_distro = self.__assessment__(train_data, self.test_set, protected_attribute,
                                                                    True)

        metrics_df = pd.concat([val_surrogate_metrics, test_surrogate_metrics, val_final_metrics, test_final_metrics])
        distribution_df = pd.concat([val_surrogate_distro, test_surrogate_distro, val_final_distro, test_final_distro])

        return metrics_df, distribution_df

    def __assessment__(self,
                       train_set: Dataset,
                       validation_set: Dataset,
                       protected_attribute: str,
                       test_assessment: bool = False) -> (pd.DataFrame, pd.DataFrame):

        def classifier_assessment_wrapper(_args: tuple) -> (pd.DataFrame, pd.DataFrame):
            _classifier, _train_data, _validation_data, _protected_attribute = _args
            _, _metrics, _distribution = classifier_assessment(_classifier, _train_data, _validation_data,
                                                               _protected_attribute)

            return _metrics, _distribution

        # Surrogate Classifiers
        pipeline_details = {
            'dataset': self.dataset.name,
            'protected_attribute': protected_attribute,
            'protected_class': 'NA',
            'unbiasing_algorithm': self.unbiasing_algorithm,
            'data': self.set,
            'classifier_type': 'Test' if test_assessment else 'Surrogate',
            'num_iterations': 0
        }

        classifiers = [self.test_classifier] if test_assessment else self.surrogate_classifiers

        args_list = [(classifier, train_set, validation_set, protected_attribute) for classifier in classifiers]
        all_metrics, all_distributions = pd.DataFrame(), pd.DataFrame()
        for args in args_list:
            metrics, distribution = classifier_assessment_wrapper(args)
            all_metrics = pd.concat([all_metrics, metrics]).reset_index(drop=True)
            all_distributions = pd.concat([all_distributions, distribution]).reset_index(drop=True)

        pipeline_details_df = dict_to_dataframe(pipeline_details)

        pipeline_metrics_df = pd.concat([pipeline_details_df] * len(all_metrics), ignore_index=True)
        pipeline_metrics_df = pd.concat([pipeline_metrics_df, all_metrics], axis=1).reset_index(drop=True)

        pipeline_distribution_df = pd.concat([pipeline_details_df] * len(all_distributions), ignore_index=True)
        pipeline_distribution_df = (pd.concat([pipeline_distribution_df, all_distributions], axis=1)
                                    .reset_index(drop=True))

        return pipeline_metrics_df, pipeline_distribution_df

    def __bias_reduction__(self, train_set: Dataset, validation_set: Dataset,
                           unbiasing_algorithm: Algorithm, protected_attribute: str) -> (pd.DataFrame, pd.DataFrame):

        unbiasing_algorithm_type = 'binary' if unbiasing_algorithm.is_binary else 'multi-class'
        logger.info(f'[INTERVENTION] Reducing bias for {unbiasing_algorithm_type} '
                    f'sensitive attribute {protected_attribute} with unbiasing algorithm'
                    f' {unbiasing_algorithm.__class__.__name__}.')

        metrics_df, distro_df = pd.DataFrame(), pd.DataFrame()
        if unbiasing_algorithm.is_binary:
            attribute_values = train_set.get_dummy_protected_feature(protected_attribute)
        else:
            attribute_values = pd.DataFrame(train_set.get_protected_feature(protected_attribute),
                                            columns=[protected_attribute])

        for value in attribute_values:
            if unbiasing_algorithm.is_binary:
                logger.info(
                    f"[INTERVENTION] Correcting bias w.r.t. attribute {protected_attribute} for {value} with "
                    f"{unbiasing_algorithm.__class__.__name__}")

            transformed_dataset = copy.deepcopy(train_set)
            transformed_dataset.set_feature(protected_attribute, attribute_values[value])

            if unbiasing_algorithm.needs_auxiliary_data:
                unbiasing_algorithm.auxiliary_data = validation_set

            for i in range(self.num_iterations):
                logger.info(f"[INTERVENTION] Iteration {i + 1} / {self.num_iterations}.")
                unbiasing_algorithm.iteration_number = i + 1
                unbiasing_algorithm.fit(transformed_dataset, protected_attribute)
                transformed_dataset = unbiasing_algorithm.transform(transformed_dataset)

                if transformed_dataset.error_flag:
                    logger.error(
                        f'[INTERVENTION] Error occurred bias correction w.r.t. attribute {protected_attribute} '
                        f'for {value} with "{unbiasing_algorithm.__class__.__name__}"')
                    continue

                metrics, distribution = self.__post_intervention__(transformed_dataset, protected_attribute)
                metrics['num_iterations'], distribution['num_iterations'] = ([i + 1] * metrics.shape[0],
                                                                             [i + 1] * distribution.shape[0])
                metrics['protected_class'], distribution['protected_class'] = ([value] * metrics.shape[0],
                                                                               [value] * distribution.shape[0])

                metrics_df, distro_df = pd.concat([metrics_df, metrics]), pd.concat([distro_df, distribution])

        return metrics_df, distro_df

    def run(self) -> None:
        self.metrics_dict.update({unbiasing_algorithm.__class__.__name__: pd.DataFrame()
                                  for unbiasing_algorithm in self.unbiasing_algorithms})
        self.distribution_dict.update({unbiasing_algorithm.__class__.__name__: pd.DataFrame()
                                       for unbiasing_algorithm in self.unbiasing_algorithms})

        try:
            logger.info("[PIPELINE] Start.")

            # split
            self.train_set, self.validation_set, self.test_set = self.dataset.split()

            # scale
            self.__scale__(MinMaxScaler())

            # pre-intervention
            pre_intervention_metrics, pre_intervention_distribution = self.__pre_intervention__(self.train_set)
            self.metrics_dict.update(
                {unbiasing_algorithm.__class__.__name__: pre_intervention_metrics
                 for unbiasing_algorithm in self.unbiasing_algorithms})
            self.distribution_dict.update(
                {unbiasing_algorithm.__class__.__name__: pre_intervention_distribution
                 for unbiasing_algorithm in self.unbiasing_algorithms})

            # correction
            for attribute, unbiasing_algorithm in list(itertools.product(self.train_set.protected_features_names,
                                                                         self.unbiasing_algorithms)):
                self.unbiasing_algorithm = unbiasing_algorithm.__class__.__name__

                attribute_metrics_df, attribute_distribution_df = self.__bias_reduction__(self.train_set,
                                                                                          self.validation_set,
                                                                                          unbiasing_algorithm,
                                                                                          attribute)

                self.metrics_dict[unbiasing_algorithm.__class__.__name__] = pd.concat(
                    [self.metrics_dict[unbiasing_algorithm.__class__.__name__], attribute_metrics_df])
                self.distribution_dict[unbiasing_algorithm.__class__.__name__] = pd.concat(
                    [self.distribution_dict[unbiasing_algorithm.__class__.__name__], attribute_distribution_df])

            logger.info("[PIPELINE] End.")

        except Exception as e:
            logger.error(f'An error occurred in the pipeline: \n {e}')
            raise

    def save(self, path: str = 'results') -> None:
        if self.metrics_dict is not None:
            for key, result in self.metrics_dict.items():
                write_dataframe_to_csv(df=round_df(result, NUM_DECIMALS), dataset_name=self.dataset.name,
                                       path=f'{path}/{key}')
            for key, result in self.distribution_dict.items():
                result = result.fillna(0.0)
                write_dataframe_to_csv(df=round_df(result, NUM_DECIMALS), dataset_name=self.dataset.name,
                                       path=f'{path}/{key}/distribution')
        else:
            raise KeyError('Results are empty!')

    def save_algorithm_results(self, algorithm: str, attribute: str, path: str = 'results') -> None:
        if self.metrics_dict is not None:
            suffix = f'_intermediary_{attribute}_'
            write_dataframe_to_csv(df=round_df(self.metrics_dict[algorithm], NUM_DECIMALS),
                                   dataset_name=self.dataset.name + suffix,
                                   path=f'{path}/{algorithm}')
        else:
            raise KeyError('Results are empty!')

    def run_and_save(self, path: str = 'results') -> None:
        self.run()
        self.save(path)

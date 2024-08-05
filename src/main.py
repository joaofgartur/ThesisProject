"""
Author: João Artur
Project: Master's Thesis
Last edited: 30-11-2023
"""
import argparse

from datasets import Dataset, GermanCredit, AdultIncome, LawSchoolAdmissions

from utils import Configs, get_seed, set_global_configs
from algorithms.utils import AlgorithmOptions, get_unbiasing_algorithm

from protocol import Pipeline
from utils import logger, extract_filename, set_seed, get_surrogate_classifiers, get_test_classifier


def load_dataset(dataset: str, configs: Configs) -> Dataset:
    match dataset:
        case "GERMAN_CREDIT":
            return GermanCredit(configs.get_dataset_configs('GERMAN_CREDIT'))
        case "ADULT_INCOME":
            return AdultIncome(configs.get_dataset_configs('ADULT_INCOME'))
        case "LAW_SCHOOL_ADMISSION":
            return LawSchoolAdmissions(configs.get_dataset_configs('LAW_SCHOOL_ADMISSION'))
        case _:
            logger.error('Dataset unknown! Currently supported datasets are: '
                         'GERMAN_CREDIT, ADULT_INCOME, LAW_SCHOOL_ADMISSION.')
            raise NotImplementedError


def run_pipeline(seed, args, configs: Configs):
    set_seed(seed)

    dataset = load_dataset(args.dataset, configs)
    surrogate_classifiers = get_surrogate_classifiers()
    test_classifier = get_test_classifier()

    if args.all:
        unbiasing_algorithms = [get_unbiasing_algorithm(j) for j in AlgorithmOptions]
    else:
        unbiasing_algorithms = [get_unbiasing_algorithm(AlgorithmOptions(args.algorithm))]

    pipeline = Pipeline(dataset, unbiasing_algorithms, surrogate_classifiers, test_classifier, configs.num_iterations)
    pipeline.run_and_save()


if __name__ == '__main__':
    logger.info(f'[{extract_filename(__file__)}] Initializing.')

    parser = argparse.ArgumentParser(description='Run the pipeline for the thesis.')
    parser.add_argument('-dataset', type=str, default='-1', help='Dataset for bias reduction.')
    parser.add_argument('-algorithm', type=int, default='-1', help='Algorithm for bias reduction.')
    parser.add_argument('-all', action='store_true', help='Run all algorithms.')
    parser.add_argument('-configs', type=str, default='-1', help='Configuration for the pipeline.')
    parser.add_argument('-seed', type=int, default='-1', help='Seed for random number generation.')

    args = parser.parse_args()

    if args.seed != -1:
        set_seed(args.seed)

    configs = Configs(args.configs)
    set_global_configs(configs)

    base_seed = get_seed()
    for i in range(configs.num_runs):
        logger.info(f'[{extract_filename(__file__)}] Starting run {i + 1}/{configs.num_runs}.')
        run_seed = base_seed + i
        run_pipeline(run_seed, args, configs)

    logger.info(f'[{extract_filename(__file__)}] End of program.')

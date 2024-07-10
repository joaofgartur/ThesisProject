"""
Author: João Artur
Project: Master's Thesis
Last edited: 30-11-2023
"""
import argparse
import configparser

from configs import (get_global_configs, load_dataset, AlgorithmOptions, load_algorithm)

from protocol import Pipeline
from helpers import logger, extract_filename, set_seed, get_seed, get_surrogate_classifiers, get_test_classifier


def run_pipeline(seed, args, dataset_configs_file, algorithms_configs_file, num_iterations):
    set_seed(seed)

    dataset = load_dataset(args.dataset, dataset_configs_file)
    surrogate_classifiers = get_surrogate_classifiers()
    test_classifier = get_test_classifier()

    if args.all:
        unbiasing_algorithms = [load_algorithm(algorithms_configs_file, j) for j in AlgorithmOptions]
    else:
        unbiasing_algorithms = [load_algorithm(algorithms_configs_file, AlgorithmOptions(args.algorithm))]

    pipeline = Pipeline(dataset, unbiasing_algorithms, surrogate_classifiers, test_classifier, num_iterations)
    pipeline.run_and_save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the pipeline for the thesis.')
    parser.add_argument('-dataset', type=str, default='-1', help='Dataset for bias reduction.')
    parser.add_argument('-algorithm', type=int, default='-1', help='Algorithm for bias reduction.')
    parser.add_argument('-all', action='store_true', help='Run all algorithms.')
    parser.add_argument('-configs', type=str, default='-1', help='Configuration for the pipeline.')
    parser.add_argument('-runs', type=int, default=1, help='Number of runs to execute.')
    parser.add_argument('-seed', type=int, default='-1', help='Seed for random number generation.')

    args = parser.parse_args()

    if args.seed != -1:
        set_seed(args.seed)

    global_configs = configparser.ConfigParser()
    dataset_cfg_file, algorithms_cfg_file, results_path, num_iters = get_global_configs(args.configs)

    logger.info(f'[{extract_filename(__file__)}] Initializing.')

    num_runs = args.runs
    base_seed = get_seed()
    for i in range(num_runs):
        logger.info(f'[{extract_filename(__file__)}] Starting run {i + 1}/{num_runs}.')
        run_seed = base_seed + i
        run_pipeline(run_seed, args, dataset_cfg_file, algorithms_cfg_file, num_iters)

    logger.info(f'[{extract_filename(__file__)}] End of program.')

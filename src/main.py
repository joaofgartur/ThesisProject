"""
Author: João Artur
Project: Master's Thesis
Last edited: 30-11-2023
"""
import argparse
import configparser

from configs import (get_global_configs, load_dataset, AlgorithmOptions, load_algorithm, surrogate_models,
                     test_classifier)

from protocol import Pipeline
from helpers import logger, extract_filename

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the pipeline for the thesis.')
    parser.add_argument('-dataset', type=str, default='-1', help='Dataset for bias reduction.')
    parser.add_argument('-algorithm', type=int, default='-1', help='Algorithm for bias reduction.')
    parser.add_argument('-all', action='store_true', help='Run all algorithms.')
    parser.add_argument('-configs', type=str, default='-1', help='Configuration for the pipeline.')
    parser.add_argument('-runs', type=int, default=1, help='Number of runs to execute.')

    args = parser.parse_args()

    global_configs = configparser.ConfigParser()
    dataset_configs_file, algorithms_configs_file, results_path, num_iterations = get_global_configs(args.configs)

    num_runs = args.runs

    logger.info(f'[{extract_filename(__file__)}] Initializing.')

    for i in range(max(1, num_runs)):
        dataset = load_dataset(args.dataset, dataset_configs_file)
        if args.all:
            unbiasing_algorithms = [load_algorithm(algorithms_configs_file, j) for j in AlgorithmOptions]
        else:
            unbiasing_algorithms = [load_algorithm(algorithms_configs_file, AlgorithmOptions(args.algorithm))]

        pipeline = Pipeline(dataset, unbiasing_algorithms, surrogate_models, test_classifier, num_iterations)
        pipeline.run_and_save()

    logger.info(f'[{extract_filename(__file__)}] End of program.')

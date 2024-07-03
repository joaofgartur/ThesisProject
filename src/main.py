"""
Author: João Artur
Project: Master's Thesis
Last edited: 30-11-2023
"""
import argparse
import configparser
from concurrent.futures import ProcessPoolExecutor, as_completed

from configs import (get_global_configs, load_dataset, AlgorithmOptions, load_algorithm, get_surrogate_classifiers, get_test_classifier)

from protocol import Pipeline
from helpers import logger, extract_filename, set_seed, get_seed


# Function to load dataset and algorithms, and run the pipeline
def run_pipeline(seed, args, dataset_configs_file, algorithms_configs_file, num_iterations):
    # Set the random seed for reproducibility
    set_seed(seed)

    # Load dataset
    dataset = load_dataset(args.dataset, dataset_configs_file)
    n = dataset.features.shape[0]
    surrogate_classifiers = get_surrogate_classifiers()
    test_classifier = get_test_classifier(n)

    # Load unbiasing algorithms
    if args.all:
        unbiasing_algorithms = [load_algorithm(algorithms_configs_file, j) for j in AlgorithmOptions]
    else:
        unbiasing_algorithms = [load_algorithm(algorithms_configs_file, AlgorithmOptions(args.algorithm))]

    # Initialize and run the pipeline
    pipeline = Pipeline(dataset, unbiasing_algorithms, surrogate_classifiers, test_classifier, num_iterations)
    pipeline.run_and_save()


# Main function to execute multiple runs in parallel
def execute_runs_with_limit(num_runs, base_seed, args, dataset_configs_file, algorithms_configs_file, surrogate_models,
                            test_classifier, num_iterations, max_workers):
    seeds = [base_seed + i for i in range(num_runs)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        # Submit all tasks (run_pipeline) asynchronously
        for seed in seeds:
            future = executor.submit(run_pipeline, seed, args, dataset_configs_file, algorithms_configs_file,
                                     surrogate_models, test_classifier, num_iterations)
            futures.append(future)

        # Wait for all futures to complete
        for future in as_completed(futures):
            try:
                result = future.result()  # Ensure all tasks are completed
                print(f"Pipeline completed with success!")
            except Exception as e:
                print(f"An error occurred: {e}")


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
    run_pipeline(get_seed(), args, dataset_configs_file, algorithms_configs_file, num_iterations)

    logger.info(f'[{extract_filename(__file__)}] Initializing.')

    # execute_runs_with_limit(num_runs, get_seed(), args, dataset_configs_file, algorithms_configs_file, surrogate_models,test_classifier, num_iterations, max_workers)

    logger.info(f'[{extract_filename(__file__)}] End of program.')

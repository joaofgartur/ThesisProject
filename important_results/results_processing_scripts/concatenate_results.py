import os

import pandas as pd


def check_missing_seeds(base_directory, base_seed, num_runs):
    algorithms = [name for name in os.listdir(base_directory)]
    for algorithm in algorithms:
        directory = f'{base_directory}/{algorithm}'
        files = [name for name in os.listdir(directory)]

        missing_seeds = []
        for i in range(base_seed, base_seed + num_runs):
            filename = f'seed_{i}_German Credit.csv'
            if filename not in files:
                missing_seeds.append(i)
        print(f'Missing seeds for {algorithm}: {missing_seeds}')


def concatenate_results(base_directory, base_seed, num_runs, algorithms, attribute_name):

    for seed in range(base_seed, base_seed + num_runs):
        try:
            seed_df = pd.DataFrame()
            for algorithm in algorithms:
                algorithm_df = pd.read_csv(f'{base_directory}/{algorithm}/seed_{seed}_German Credit.csv')
                algorithm_df = algorithm_df[algorithm_df['attribute'] == attribute_name]

                if algorithm != 'Massaging':
                    rows_to_drop = algorithm_df[algorithm_df['num_iterations'] == 0].index
                    algorithm_df = algorithm_df.drop(rows_to_drop)

                seed_df = pd.concat([seed_df, algorithm_df], ignore_index=True)

            seed_df.to_csv(f'{base_directory}/German Credit_{seed}_{attribute_name}.csv', index=False)
        except FileNotFoundError:
            continue


def check_shape(base_directory, base_seed, num_runs):
    initial_shape = None
    all_equal = True
    for seed in range(base_seed, base_seed + num_runs):
        try:
            df = pd.read_csv(f'{base_directory}/seed_{seed}_German Credit.csv')
            if seed == base_seed:
                initial_shape = df.shape
            else:
                equal = df.shape == initial_shape
                all_equal = all_equal and equal
                if not equal:
                    print(f'Seed: {seed}, Initial Shape: {initial_shape}, Shape: {df.shape}, Equal: {equal}')
        except FileNotFoundError:
            continue

    print(f'All equal: {all_equal}')


def handle_concatenation(base_directory, base_seed, num_runs, algorithms, attribute_name):

    print(f'Handling {base_directory} for {attribute_name}')
    check_missing_seeds(base_directory, base_seed, num_runs)
    concatenate_results(base_directory, base_seed, num_runs, algorithms, attribute_name)
    check_shape(base_directory, base_seed, num_runs)


if __name__ == '__main__':

    base_seed = 42
    num_runs = 30
    base_directory = 'metrics'
    algorithms = ['Massaging', 'Reweighing', 'DisparateImpactRemover',
                  'LexicographicGeneticAlgorithmFairFeatureSelection',
                  'MulticlassLexicographicGeneticAlgorithmFairFeatureSelection', 'FairGenes']

    handle_concatenation('metrics', base_seed, num_runs, algorithms, 'Attribute9')

    handle_concatenation('distributions', base_seed, num_runs, algorithms, 'Attribute9')

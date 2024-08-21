import pandas as pd


def correct_mapping(df):
    mapping = {
        'Massaging': 'MSG',
        'Reweighing': 'REW',
        'Disparate Impact Remover': 'DIR',
        'LexicographicGeneticAlgorithmFairFeatureSelection': 'LGAFFS'
    }

    def correct_algorithm_name(genotype):
        for key, value in mapping.items():
            if key in genotype:
                genotype = genotype.replace(key, value)

        return genotype

    df['Genotype'] = df['Genotype'].apply(correct_algorithm_name)

def correct_best_individuals(base_directory, base_seed, num_runs, dataset, attribute):

    for seed in range(base_seed, base_seed + num_runs):
        seed_df = pd.DataFrame()
        try:
            seed_directory = f'seed_{seed}_FairGenes_{attribute}_fitness_evolution.csv'
            first_iteration_df = pd.read_csv(f'{base_directory}/{dataset}/1_iteration/{seed_directory}')
            second_iteration_df = pd.read_csv(f'{base_directory}/{dataset}/2_iteration/{seed_directory}')

            seed_df = pd.concat([seed_df, first_iteration_df, second_iteration_df], ignore_index=True)
            correct_mapping(seed_df)

            seed_df.to_csv(f'{base_directory}/{dataset}/FairGenes_{attribute}_fitness_evolution_seed_{seed}.csv', index=False)
        except FileNotFoundError:
            print(f'Seed {seed} not found.')

if __name__ == '__main__':

    base_seed = 42
    num_runs = 30
    dataset = 'German Credit'
    base_directory = 'best_individuals'

    correct_best_individuals(base_directory, base_seed, num_runs, dataset, 'Attribute9')



import pandas as pd


def compute_sum_square_distances(df):
    columns = df.columns.unique()

    result = pd.DataFrame()
    for col in columns:
        metric_columns = df[col]
        sum_square_distances = ((metric_columns.sub(1.0, axis=0)) ** 2).sum(axis=1)
        result[col] = sum_square_distances

    return result


def compute_average(df):
    unique_columns = df.columns.unique()
    result = pd.DataFrame()
    for col in unique_columns:
        metric_columns = df[col]
        average = metric_columns.mean(axis=1)
        result[col] = average

    return result


def compute_agregated_metrics(directory, dataset, attribute, base_seed, num_runs):

    first_file = f'{directory}/{dataset}_{base_seed}_{attribute}.csv'
    final_df = pd.read_csv(first_file, index_col=0)

    fairness_columns = [col for col in final_df.columns if 'fairness' in col]
    performance_columns = [col for col in final_df.columns if 'performance' in col]

    final_df = final_df.drop(columns=fairness_columns + performance_columns)
    fairness_df = pd.DataFrame()
    performance_df = pd.DataFrame()

    for seed in range(base_seed, base_seed + num_runs):
        try:
            seed_df = pd.read_csv(f'{directory}/{dataset}_{seed}_{attribute}.csv', index_col=0)
            fairness_df = pd.concat([fairness_df, seed_df[seed_df.columns.intersection(fairness_columns)]], axis=1)
            performance_df = pd.concat([performance_df, seed_df[seed_df.columns.intersection(performance_columns)]], axis=1)
        except FileNotFoundError:
            continue

    fairness_df = compute_sum_square_distances(fairness_df)
    performance_df = compute_average(performance_df)

    final_df = pd.concat([final_df, fairness_df, performance_df], axis=1)

    final_df.to_csv(f'{directory}/{dataset}_{attribute}.csv')


if __name__ == '__main__':
    directory = 'metrics'
    base_seed = 42
    num_runs = 30
    dataset = 'Law School Admission Bar Passage'
    attribute = 'race1'

    compute_agregated_metrics(directory, dataset, attribute, base_seed, num_runs)

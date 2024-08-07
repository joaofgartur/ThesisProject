
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

    """
    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, index_col=0)

            # Store values for each numeric column
            for col in col_values.keys():
                col_values[col] = pd.concat([col_values[col], df[col]], axis=1)

    for col in col_values.keys():
        # mean_values = np.round(col_values[col].mean(axis=1, skipna=True), decimals=4)
        sum_square_distances = np.round(((col_values[col].sub(1.0, axis=0)) ** 2).sum(axis=1), decimals=4)

        # Create a DataFrame with mean values and sum of square distances
        new_df = pd.DataFrame({
            # f'{col}_mean': mean_values,
            f'{col}_ssd': sum_square_distances
        }, index=final_df.index)

        # Get the index position of the current column in final_df
        index_position = final_df.columns.get_loc(col)

        # Drop the original column from final_df
        final_df = final_df.drop(columns=col)

        # Concatenate new_df with final_df at the corresponding index position
        final_df = pd.concat([final_df.iloc[:, :index_position], new_df, final_df.iloc[:, index_position:]], axis=1)

    final_df.to_csv(output_file)
    """


if __name__ == '__main__':
    directory = 'metrics'
    base_seed = 42
    num_runs = 30

    compute_agregated_metrics(directory, 'German Credit', 'Attribute9', base_seed, num_runs)

import os

import numpy as np
import pandas as pd


def compute_agregated_metrics(folder_path, output_file):
    # Find the first CSV file in the folder
    first_file_path = None
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            first_file_path = os.path.join(folder_path, filename)
            break
    if first_file_path is None:
        print("No CSV files found in the folder.")
        return

    final_df = pd.read_csv(first_file_path, index_col=0)

    # Initialize a dictionary to store the values for each numeric column
    col_values = {col: pd.DataFrame() for col in final_df.select_dtypes(include='number').columns}

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


if __name__ == '__main__':
    algorithms = ['Massaging', 'Reweighing', 'DisparateImpactRemover', 'AIF360LearningFairRepresentations', 'LGAFFS', 'PermutationGeneticAlgorithm']

    for algorithm in algorithms:
        input_folder = f"results_german_28_04_2024/{algorithm}"
        output_file = f"{input_folder}/{algorithm}_mean_values.csv"
        compute_agregated_metrics(input_folder, output_file)

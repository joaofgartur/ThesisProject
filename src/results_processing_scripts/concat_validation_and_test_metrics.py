import os

import pandas as pd
import glob


def concatenate_groups(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Replace 'data_column' with the name of the column you want to group by
    grouped_data = df.groupby('data')

    # Create a folder to store the temporary files if it doesn't exist
    output_folder = 'temp_files'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Write each group into different files
    for group_name, group_data in grouped_data:
        filename = os.path.join(output_folder,
                                f"{group_name}.csv")  # Constructing filename dynamically with group name as prefix
        group_data.to_csv(filename, index=False)  # Write the group data to CSV without index
        print(f"Group '{group_name}' has been written to '{filename}'.")

    csv_files = glob.glob('temp_files/*.csv')

    file1 = pd.read_csv(csv_files[1])
    file2 = pd.read_csv(csv_files[0])

    file1_prefixed = file1.add_suffix('_' + csv_files[1].split('.')[0])
    file2_prefixed = file2.add_suffix('_' + csv_files[0].split('.')[0])

    concatenated_data = pd.concat([file1_prefixed, file2_prefixed], axis=1)

    # Group columns by their prefixes
    column_groups = {}
    for column in concatenated_data.columns:
        prefix = column.split('_')[0]  # Assuming the prefix is before the first underscore
        if prefix not in column_groups:
            column_groups[prefix] = []
        column_groups[prefix].append(column)

    # Create a new DataFrame with reordered columns
    reordered_data = pd.DataFrame()
    for prefix, columns in column_groups.items():
        reordered_data = pd.concat([reordered_data, concatenated_data[columns]], axis=1)

    reordered_data.to_csv(output_file, index=False)


if __name__ == '__main__':
    algorithms = ['Massaging', 'Reweighing', 'DisparateImpactRemover', 'AIF360LearningFairRepresentations', 'LGAFFS',
                  'PermutationGeneticAlgorithm']

    for algorithm in algorithms:
        input_folder = f"results_german_28_04_2024/{algorithm}"
        input_file = f"{input_folder}/{algorithm}_mean_values.csv"
        output_file = f"{input_folder}/concatenated_{algorithm}.csv"
        concatenate_groups(input_file, output_file)

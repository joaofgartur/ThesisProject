import pandas as pd

DI_SSE_LIMIT = 1.2
FAIRNESS_SSE_LIMIT = 0.075

"""
### Aggregate metrics computation
"""


def compute_sum_square_error(df):
    metrics = df.columns.unique()

    result = pd.DataFrame()
    for col in metrics:
        metric_columns = df[col]
        sum_square_distances = ((metric_columns.sub(1.0, axis=0)) ** 2).sum(axis=1)
        result[col] = sum_square_distances

    return result


def compute_agregated_metrics(directory, dataset, attribute, base_seed, num_runs, num_decimals=3):
    # get the columns
    first_file = f'{directory}/{dataset}_{base_seed}_{attribute}.csv'
    final_df = pd.read_csv(first_file, index_col=0)

    # get fairness and performance columns
    fairness_columns = [col for col in final_df.columns if 'fairness' in col]
    performance_columns = [col for col in final_df.columns if 'performance' in col]

    # keep metadata columns
    final_df = final_df.drop(columns=fairness_columns + performance_columns)
    fairness_df = pd.DataFrame()
    performance_df = pd.DataFrame()

    # compute aggregated metrics
    for seed in range(base_seed, base_seed + num_runs):
        try:
            seed_df = pd.read_csv(f'{directory}/{dataset}_{seed}_{attribute}.csv', index_col=0)
            fairness_df = pd.concat([fairness_df, seed_df[seed_df.columns.intersection(fairness_columns)]], axis=1)
            performance_df = pd.concat([performance_df, seed_df[seed_df.columns.intersection(performance_columns)]],
                                       axis=1)
        except FileNotFoundError:
            continue

    fairness_df = compute_average(fairness_df)
    performance_df = compute_average(performance_df)

    final_df = pd.concat([final_df, fairness_df, performance_df], axis=1)
    final_df = final_df.round(num_decimals)

    return final_df


def compute_average(df):
    unique_columns = df.columns.unique()
    result = pd.DataFrame()
    for col in unique_columns:
        metric_columns = df[col]
        average = metric_columns.mean(axis=1)
        result[col] = average

    return result


"""
### Base tables computation
"""


def pad_decimals(df, num_decimals):
    def format_float(x):
        if isinstance(x, (float)):
            return f"{x:.{num_decimals}f}"
        return x

    df = df.applymap(format_float)

    return df


def parse_global_df(global_df):
    binary_algorithms = ['Massaging', 'Reweighing', 'DisparateImpactRemover',
                         'LexicographicGeneticAlgorithmFairFeatureSelection']
    multiclass_algorithms = ['MulticlassLexicographicGeneticAlgorithmFairFeatureSelection', 'FairGenes']

    first_part = global_df[global_df[iterations_column] == 0]
    first_part.loc[:, 'group'] = first_part['value']
    binary_part = global_df[
        (global_df['algorithm'].isin(binary_algorithms)) & (global_df['group'] == global_df['value'])]
    multiclass_part = global_df[(global_df['algorithm'].isin(multiclass_algorithms))]

    global_df = pd.concat([first_part, binary_part, multiclass_part])

    return global_df


def parse_classifiers_df(global_df, exclude_validation=False):
    classifiers = global_df['classification_algorithm'].unique()

    # print(global_df.head().to_string())

    for classifier in classifiers:
        classifier_table = global_df[global_df['classification_algorithm'] == classifier]
        classifier_table = classifier_table.drop(columns=['classification_algorithm'])

        if exclude_validation:
            classifier_table = classifier_table[classifier_table['set'] != 'Validation']

        classifier_table = classifier_table.sort_values(by=['value'], kind='mergesort')
        classifier_table.to_csv(f'{directory}/{dataset}_{attribute}_{classifier}.csv')


if __name__ == '__main__':

    dataset = 'Law School Admission Bar Passage'
    attribute = 'race1'

    iterations_column = 'iteration' if dataset == 'German Credit' else 'iterations'

    directory = 'metrics'
    base_seed = 42
    num_runs = 30
    classifiers = ['LogisticRegression', 'SVC', 'GaussianNB', 'DecisionTreeClassifier', 'RandomForestClassifier',
                   'XGBClassifier']

    aggregated_metrics = compute_agregated_metrics(directory, dataset, attribute, base_seed, num_runs)
    aggregated_metrics.to_csv(f'{directory}/{dataset}_{attribute}_aggregated.csv')
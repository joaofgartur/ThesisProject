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
            performance_df = pd.concat([performance_df, seed_df[seed_df.columns.intersection(performance_columns)]], axis=1)
        except FileNotFoundError:
            continue

    fairness_df = compute_sum_square_error(fairness_df)
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
    binary_algorithms = ['Massaging', 'Reweighing', 'DisparateImpactRemover', 'LexicographicGeneticAlgorithmFairFeatureSelection']
    multiclass_algorithms = ['MulticlassLexicographicGeneticAlgorithmFairFeatureSelection', 'FairGenes']

    first_part = global_df[global_df[iterations_column] == 0]
    first_part.loc[:, 'group'] = first_part['value']
    binary_part = global_df[(global_df['algorithm'].isin(binary_algorithms)) & (global_df['group'] == global_df['value'])]
    multiclass_part = global_df[(global_df['algorithm'].isin(multiclass_algorithms))]

    global_df = pd.concat([first_part, binary_part, multiclass_part])

    return global_df


def parse_classifiers_df(global_df, exclude_validation=False):

    classifiers = global_df['classification_algorithm'].unique()

    for classifier in classifiers:
        classifier_table = global_df[global_df['classification_algorithm'] == classifier]
        classifier_table = classifier_table.drop(columns=['classification_algorithm'])

        if exclude_validation:
            classifier_table = classifier_table[classifier_table['set'] != 'Validation']

        classifier_table = classifier_table.sort_values(by=['value'], kind='mergesort')
        classifier_table.to_csv(f'{directory}/{dataset}_{attribute}_{classifier}.csv')


def color_fairness_value(value, baseline, is_di):
    formatted_val = f"{value:.3f}"

    if is_di:
        threshold = DI_SSE_LIMIT
    else:
        threshold = FAIRNESS_SSE_LIMIT

    if value <= threshold:
        return f"\\green {formatted_val}"
    elif value >= baseline:
        return f"\\red {formatted_val}"
    else:
        return f"\\yellow {formatted_val}"


def color_performance_value(value, baseline):
    formatted_val = f"{value:.3f}"
    if value > baseline:
        return f"\\green {formatted_val}"
    elif value < baseline:
        return f"\\red {formatted_val}"
    else:
        return f"\\yellow {formatted_val}"


def colour_analysis(directory, dataset, attribute, classifier):
    df = pd.read_csv(f'{directory}/{dataset}_{attribute}_{classifier}.csv', index_col=0)
    protected_groups = df['value'].unique()

    colored_df = pd.DataFrame()
    for value in protected_groups:
        baseline = df[(df['value'] == value) & (df[iterations_column] == 0)].iloc[0]

        rows_to_color = df[(df['value'] == value) & (df[iterations_column] > 0)].copy()

        fairness_columns = [col for col in rows_to_color.columns if 'fairness' in col]
        performance_columns = [col for col in rows_to_color.columns if 'performance' in col]

        for col in fairness_columns:
            is_di = 'disparate_impact' in col
            rows_to_color[col] = rows_to_color[col].apply(lambda x: color_fairness_value(x, baseline[col], is_di=is_di))

        for col in performance_columns:
            rows_to_color[col] = rows_to_color[col].apply(lambda x: color_performance_value(x, baseline[col]))

        colored_df = pd.concat([colored_df, baseline.to_frame().T, rows_to_color])

    relevant_columns = [col for col in df.columns if 'fairness' in col or 'performance' in col]
    colored_df = colored_df[relevant_columns]

    latex_table = colored_df.to_latex(index=False,
                              column_format='|c|c|c|c|c|c|c|c|c|' + 'r|' * (df.shape[1] - 9), float_format="%.3f")

    with open(f'{directory}/table_{dataset}_{attribute}_{classifier}.tex', 'w') as file:
        file.write(latex_table)


def multiple_iterations_analysis(directory, dataset, attribute, classifiers):
    df = pd.DataFrame()

    for classifier in classifiers:
        classifier_df = pd.read_csv(f'{directory}/{dataset}_{attribute}_{classifier}.csv', index_col=0)
        classifier_ = pd.DataFrame({'classifier': classifier for _ in range(classifier_df.shape[0])}, index=classifier_df.index)
        classifier_df = pd.concat([classifier_, classifier_df], axis=1)
        df = pd.concat([df, classifier_df])
    df = df[df[iterations_column] > 0]

    results = pd.DataFrame()
    protected_groups = df['value'].unique()
    algorithms = df['algorithm'].unique()

    combinations = [(classifier, algorithm, value) for classifier in classifiers for algorithm in algorithms for value in protected_groups]

    for comb in combinations:
        classifier, algorithm, value = comb
        algorithm_df = df[(df['classifier'] == classifier) & (df['algorithm'] == algorithm) & (df['value'] == value)]
        try:
            first_iteration = algorithm_df[algorithm_df[iterations_column] == 1].iloc[0]
            second_iteration = algorithm_df[algorithm_df[iterations_column] == 2].iloc[0]

            first_iter_fairness = first_iteration[[col for col in first_iteration.index if 'fairness' in col]]
            second_iter_fairness = second_iteration[[col for col in second_iteration.index if 'fairness' in col]]

            fairness_variation = second_iter_fairness - first_iter_fairness

            num_increases = (fairness_variation > 0).sum()
            num_decreases = (fairness_variation < 0).sum()
            num_no_change = (fairness_variation == 0).sum()
            algorithm_df = pd.DataFrame({'classifier': classifier, 'algorithm': algorithm, 'group': value,
                                         'num_increases': num_increases, 'num_decreases': num_decreases,
                                         'num_no_change': num_no_change}, index=[0])
            results = pd.concat([results, algorithm_df])

        except IndexError as e:
            continue

    total_increases = results['num_increases'].sum()
    total_decreases = results['num_decreases'].sum()
    total_no_change = results['num_no_change'].sum()

    results = pd.concat([results, pd.DataFrame({'algorithm': 'Total', 'num_increases': total_increases, 'num_decreases': total_decreases, 'num_no_change': total_no_change}, index=[0])])

    numerics = ['int16', 'int32', 'int64']
    results = results.sort_values(by=['group'], kind='mergesort')
    results = results.select_dtypes(include=numerics)

    latex_table = results.to_latex(index=False, float_format="%.3f")
    with open(f'{directory}/table_{dataset}_{attribute}_multiple_iterations.tex', 'w') as file:
        file.write(latex_table)


if __name__ == '__main__':
    
    # dataset = 'Law School Admission Bar Passage'
    # attribute = 'race1'
    dataset = 'German Credit'
    attribute = 'Attribute9'
    iterations_column = 'iteration' if dataset == 'German Credit' else 'iterations'


    directory = 'metrics'
    base_seed = 42
    num_runs = 30
    classifiers = ['LogisticRegression', 'SVC', 'GaussianNB', 'DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier']

    aggregated_metrics = compute_agregated_metrics(directory, dataset, attribute, base_seed, num_runs)

    global_df = parse_global_df(aggregated_metrics)
    global_df = pad_decimals(global_df, 3)
    parse_classifiers_df(global_df, True)

    for classifier in classifiers:
        colour_analysis(directory, dataset, attribute, classifier)

    multiple_iterations_analysis(directory, dataset, attribute, classifiers)
import pandas as pd

DI_SSE_LIMIT = 1.2
FAIRNESS_SSE_LIMIT = 0.075


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
        baseline = df[(df['value'] == value) & (df['iteration'] == 0)].iloc[0]

        rows_to_color = df[(df['value'] == value) & (df['iteration'] > 0)].copy()

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

if __name__ == '__main__':

    dataset = 'German Credit'
    attribute = 'Attribute9'
    directory = 'metrics'
    classifiers = ['LogisticRegression', 'SVC', 'GaussianNB', 'DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier']

    for classifier in classifiers:
        colour_analysis(directory, dataset, attribute, classifier)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

for patch in plt.gca().patches:
    current_width = patch.get_width()
    patch.set_width(current_width * 0.8)  # Reduce the width by 20%
    patch.set_x(patch.get_x() + current_width * 0.1)  # Reposition the bars slightly

palette_collection = "pastel"
label_font_size = 14
title_font_size = 16
ticks_font_size = 12
font_name = 'Times New Roman'
show_plot = True

def line_plot(df, x_axis, y_axis, title, hue=None, style=None):
    plt.figure(figsize=(12, 8))

    palette = sns.color_palette(palette_collection, len(df))

    sns.lineplot(
        x=x_axis,
        y=y_axis,
        data=df,
        palette=palette,
        hue=hue if hue else None,
        style=style if style else None
    )

    plt.xlabel(x_axis, fontsize=label_font_size, fontname=font_name)
    plt.ylabel(y_axis, fontsize=label_font_size, fontname=font_name)
    plt.title(title, fontsize=label_font_size, fontweight='bold', fontname=font_name)

    plt.xticks(fontsize=ticks_font_size, rotation=45)
    plt.yticks(fontsize=ticks_font_size)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    sns.despine()

    plt.savefig(f'{title}.svg', format='svg', dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()

def cat_plot(df, x_axis, y_axis, title, hue=None):
    plt.figure(figsize=(12, 8))

    palette = sns.color_palette(palette_collection, len(df))

    sns.catplot(
        x=x_axis,
        y=y_axis,
        data=df,
        palette=palette,
        edgecolor='black',
        dodge=True,
        kind='bar',
        alpha=0.7,
        hue=hue if hue else None
    )

    plt.xlabel(x_axis, fontsize=label_font_size, fontname=font_name)
    plt.ylabel(y_axis, fontsize=label_font_size, fontname=font_name)
    plt.title(title, fontsize=label_font_size, fontweight='bold', fontname=font_name)

    plt.xticks(fontsize=ticks_font_size, rotation=45)
    plt.yticks(fontsize=ticks_font_size)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    sns.despine()

    plt.savefig(f'{title}.svg', format='svg', dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()

def bar_plot(df, x_axis, y_axis, title, hue=None):
    plt.figure(figsize=(12, 8))

    palette = sns.color_palette(palette_collection, len(df))

    sns.barplot(
        x=x_axis,
        y=y_axis,
        data=df,
        palette=palette,
        edgecolor='black',
        dodge=True,
        hue=hue if hue else None
    )

    plt.xlabel(x_axis, fontsize=label_font_size, fontname=font_name)
    plt.ylabel(y_axis, fontsize=label_font_size, fontname=font_name)
    plt.title(title, fontsize=label_font_size, fontweight='bold', fontname=font_name)

    plt.xticks(fontsize=ticks_font_size, rotation=45)
    plt.yticks(fontsize=ticks_font_size)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    sns.despine()

    plt.savefig(f'{title}.svg', format='svg', dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()

def compute_global_df(base_directory, base_seed, num_runs, dataset, attribute, head_size=0, tail_size=0):

    global_df = pd.DataFrame()
    for seed in range(base_seed, base_seed + num_runs):
        try:
            file = f'{base_directory}/{dataset}/FairGenes_{attribute}_fitness_evolution_seed_{seed}.csv'
            seed_df = pd.read_csv(file)

            if head_size > 0:
                seed_df = seed_df.head(head_size)

            if tail_size > 0:
                seed_df = seed_df.tail(tail_size)

            global_df = pd.concat([global_df, seed_df], ignore_index=True)
        except FileNotFoundError:
            print(f'Seed {seed} not found.')

    return global_df


def map_genotypes(df):
    mapping = pd.factorize(df['Genotype'])[0]
    mapping = pd.concat([pd.Series(mapping, name='Mapping'), df['Genotype']], axis=1)
    df['Genotype'] = mapping['Mapping']

    return mapping, df


def frequency_analysis(global_df):

    frequency = {}
    for row in global_df.iterrows():
        genotype = row[1]['Genotype']
        frequency[genotype] = frequency.get(genotype, 0) + 1

    frequency = {k: v for k, v in sorted(frequency.items(), key=lambda item: item[1], reverse=True)}
    frequency_df = pd.DataFrame(frequency.items(), columns=['Genotype', 'Frequency'])

    return frequency_df


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


def compute_agregated_metrics(directory, dataset, attribute, base_seed, num_runs, head_size=0):

    first_file = f'{directory}/{dataset}/FairGenes_{attribute}_fitness_evolution_seed_{base_seed}.csv'
    final_df = pd.read_csv(first_file, index_col=0)

    fairness_columns = [col for col in final_df.columns if 'fairness' in col]
    performance_columns = [col for col in final_df.columns if 'performance' in col]

    final_df = pd.DataFrame([i for i in range(1, head_size + 1)], columns=['Generation'])
    fairness_df = pd.DataFrame()
    performance_df = pd.DataFrame()

    for seed in range(base_seed, base_seed + num_runs):
        try:
            file = f'{directory}/{dataset}/FairGenes_{attribute}_fitness_evolution_seed_{seed}.csv'
            seed_df = pd.read_csv(file, index_col=0)

            if head_size > 0:
                seed_df = seed_df.head(head_size)

            seed_df = seed_df.reset_index(drop=True)
            fairness_df = pd.concat([fairness_df, seed_df[seed_df.columns.intersection(fairness_columns)]], axis=1)
            performance_df = pd.concat([performance_df, seed_df[seed_df.columns.intersection(performance_columns)]], axis=1)
        except FileNotFoundError:
            continue


    fairness_df = compute_sum_square_distances(fairness_df)
    performance_df = compute_average(performance_df)

    final_df = pd.concat([final_df, fairness_df, performance_df], axis=1)
    final_df = final_df.set_index('Generation')

    final_df.to_csv(f'{directory}/{dataset}_{attribute}.csv')

    return final_df


def best_vs_population_analysis(global_df, classifiers, metrics, metric_type='fairness'):
    metrics_df = {}

    classifier_mapping = {
        'LogisticRegression': 'LR',
        'SVC': 'SVM',
        'GaussianNB': 'NB',
        'DecisionTreeClassifier': 'DT',
        'RandomForestClassifier': 'RF'
    }

    for metric in metrics:
        metric_df = pd.DataFrame()

        for classifier in classifiers:
            best_df = global_df[[f'{classifier}_{metric_type}_{metric}']].rename(columns={f'{classifier}_{metric_type}_{metric}': f'Best_{classifier_mapping[classifier]}'})
            mean_df = global_df[[f'{classifier}_{metric_type}_{metric}mean']].rename(columns={f'{classifier}_{metric_type}_{metric}mean': f'Population_{classifier_mapping[classifier]}'})

            mean_df = mean_df.reset_index(drop=True)
            best_df = best_df.reset_index(drop=True)

            metric_df = pd.concat([metric_df, best_df, mean_df], axis=1)
            # metric_df = metric_df.set_index('Generation')
            metric_df.reset_index(drop=True, inplace=True)
            generation = range(len(metric_df))
            metric_df['Generation'] = generation
            metric_df.set_index('Generation', inplace=True)

        metrics_df[metric] = metric_df

    return metrics_df


if __name__ == '__main__':

    base_seed = 42
    num_runs = 30
    dataset = 'Law School Admission Bar Passage'
    attribute = 'race1'
    base_directory = 'best_individuals'
    classifiers = ['LogisticRegression', 'SVC', 'GaussianNB', 'DecisionTreeClassifier', 'RandomForestClassifier']
    metrics = ['disparate_impact', 'discrimination_score', 'true_positive_rate_diff', 'false_positive_rate_diff',
               'false_positive_error_rate_balance_score', 'false_negative_error_rate_balance_score', 'consistency']
    head_size = 1 if 'German' in dataset else 0
    tail_size = 1 if 'Law' in dataset else 0
    line = True if 'Law' in dataset else False

    show_plot = False

    global_df = compute_global_df(base_directory, base_seed, num_runs, dataset, attribute, head_size=head_size, tail_size=tail_size)
    frequency_df = frequency_analysis(global_df)
    mapped_genotypes, frequency_df = map_genotypes(frequency_df)
    mapped_genotypes.to_latex(f'{dataset}_mapped_genotypes.tex', index_names=False, index=False)
    bar_plot(frequency_df, title=f'Genotype Frequency Analysis for {dataset}', x_axis='Genotype', y_axis='Frequency')


    aggregated_df = compute_agregated_metrics(base_directory, dataset, attribute, base_seed, num_runs, head_size)
    metrics_dfs = best_vs_population_analysis(aggregated_df, classifiers, metrics)

    for metric, df in metrics_dfs.items():
        df_melted = df.melt(var_name='Classifier', value_name='Performance')
        df_melted['Type'] = df_melted['Classifier'].apply(lambda x: 'Best' if 'Best' in x else 'Population')
        df_melted['Classifier'] = df_melted['Classifier'].apply(
            lambda x: x.replace('Best_', '').replace('Population_', ''))

        if line:
            df_melted['Generation'] = df_melted.index % 30
            line_plot(df_melted, title=f'{dataset} - {metric}', x_axis='Generation', y_axis='Performance', hue='Classifier', style='Type')
        else:
            cat_plot(df_melted, title=f'{dataset} - {metric}', x_axis='Classifier', y_axis='Performance', hue='Type')


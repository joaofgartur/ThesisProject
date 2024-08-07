import pandas as pd


def pad_decimals(df, num_decimals):

    def format_float(x):
        if isinstance(x, (float)):
            return f"{x:.{num_decimals}f}"
        return x

    df = df.applymap(format_float)

    return df


def create_global_table(directory, dataset, attribute):
    binary_algorithms = ['Massaging', 'Reweighing', 'DisparateImpactRemover', 'LexicographicGeneticAlgorithmFairFeatureSelection']
    multiclass_algorithms = ['MulticlassLexicographicGeneticAlgorithmFairFeatureSelection', 'FairGenes']

    global_table = pd.read_csv(f'{directory}/{dataset}_{attribute}.csv', index_col=0)

    first_part = global_table[global_table['iteration'] == 0]
    binary_part = global_table[(global_table['algorithm'].isin(binary_algorithms)) & (global_table['group'] == global_table['value'])]
    multiclass_part = global_table[(global_table['algorithm'].isin(multiclass_algorithms))]

    global_table = pd.concat([first_part, binary_part, multiclass_part])

    return global_table


def create_classifiers_tables(global_table, exclude_validation=False):



    classifiers = global_table['classification_algorithm'].unique()

    for classifier in classifiers:
        classifier_table = global_table[global_table['classification_algorithm'] == classifier]
        classifier_table = classifier_table.drop(columns=['classification_algorithm'])
        if exclude_validation:
            classifier_table = classifier_table[classifier_table['set'] != 'Validation']
        classifier_table = classifier_table.sort_values(by=['value'], kind='mergesort')



        classifier_table.to_csv(f'{directory}/{dataset}_{attribute}_{classifier}.csv')


if __name__ == '__main__':

    dataset = 'German Credit'
    attribute = 'Attribute9'
    directory = 'metrics'
    exclude_validation = True

    global_table = create_global_table(directory, dataset, attribute)
    global_table = pad_decimals(global_table, 4)
    create_classifiers_tables(global_table, exclude_validation)

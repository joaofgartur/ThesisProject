from datasets import GermanCredit, AdultIncome, Compas, Dataset
from metrics import compute_metrics_suite
from algorithms import massaging


def bias_correction(dataset: Dataset, learning_settings: dict):
    # pre-correction diagnostics stage

    pre_diagnostics = {}
    for sensitive_attribute in dataset.sensitive_attributes_info.keys():
        metrics = compute_metrics_suite(dataset, sensitive_attribute)
        pre_diagnostics.update({sensitive_attribute: metrics})
    print(pre_diagnostics)

    # correction stage
    results = {}
    for sensitive_attribute in dataset.sensitive_attributes_info.keys():
        new_dataset = massaging(dataset=dataset, sensitive_attribute=sensitive_attribute,
                                learning_settings=learning_settings)
        metrics = compute_metrics_suite(new_dataset, sensitive_attribute)
        results.update({sensitive_attribute: metrics})

    # post-correction diagnostics stage
    print(results)

    # classifier training stage


if __name__ == '__main__':
    _learning_settings = {"train_size": 0.7, "test_size": 0.3, "seed": 125}

    # german_credit = GermanCredit("german", ["Attribute9"])
    # print(disparate_impact(german_credit))

    # adult_income = AdultIncome("adult", ["race"])
    # print(disparate_impact(adult_income))

    compas_info = {
        "dataset_name": "compas",
        "sensitive_attributes": {
            "race": {
                "values": ["African-American", 'Asian', 'Caucasian', 'Hispanic', 'Native American', 'Other'],
                "unprivileged_value": "African-American",
            },
            "sex": {
                "values": ["Female", "Male"],
                "unprivileged_value": "Male",
            }
        }
    }

    compas = Compas(compas_info)

    bias_correction(compas, _learning_settings)

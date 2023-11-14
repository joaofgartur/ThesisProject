from classifiers.classifiers import train_all_classifiers
from datasets import Dataset
from metrics import compute_metrics_suite


def diagnostics(dataset: Dataset, learning_settings: dict) -> dict:
    metrics_results = {}
    for sensitive_attribute in dataset.sensitive_attributes_info.keys():
        metrics = compute_metrics_suite(dataset, sensitive_attribute)
        metrics_results.update({sensitive_attribute: metrics})

    classifiers_results = train_all_classifiers(dataset, learning_settings)

    return {"metrics": metrics_results, "classifiers": classifiers_results}


def post_pre_correction_diagnostics(pre_correction_diagnostics: dict, dataset: Dataset):
    raise NotImplementedError

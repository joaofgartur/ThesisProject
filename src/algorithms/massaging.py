from datasets import Dataset
from constants import PRIVILEGED, UNPRIVILEGED, NEGATIVE_OUTCOME, POSITIVE_OUTCOME
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd


def compute_class_probabilities(dataset: Dataset, learning_settings):
    x_train, x_test, y_train, y_test = train_test_split(dataset.features, dataset.targets,
                                                        test_size=learning_settings["test_size"],
                                                        train_size=learning_settings["train_size"])

    model = GaussianNB()
    model.fit(x_train, y_train)

    # Predict Output
    class_probabilities = model.predict_proba(dataset.features)

    return class_probabilities


def ranking(dataset: Dataset, sensitive_attribute, learning_settings):
    class_probabilities = compute_class_probabilities(dataset, learning_settings)
    positive_class_probabilities = class_probabilities[:, 0]

    data = pd.concat([dataset.features, dataset.targets], axis="columns")
    label_column_name = dataset.targets.columns[0]

    # select candidates for promotion
    pr_candidates_indexes = data.index[
        (data[sensitive_attribute] == UNPRIVILEGED) & (data[label_column_name] == NEGATIVE_OUTCOME)].tolist()
    promotion_candidates = pd.DataFrame({"index": data.index, "class_probability": positive_class_probabilities})
    promotion_candidates = promotion_candidates.iloc[pr_candidates_indexes].sort_values(by='class_probability',
                                                                                        ascending=False)

    # select candidates for demotion
    dem_candidates_indexes = data.index[
        (data[sensitive_attribute] == PRIVILEGED) & (data[label_column_name] == POSITIVE_OUTCOME)].tolist()
    demotion_candidates = pd.DataFrame({"index": data.index, "class_probability": positive_class_probabilities})
    demotion_candidates = demotion_candidates.iloc[dem_candidates_indexes].sort_values(by='class_probability')

    return promotion_candidates, demotion_candidates


def massaging(dataset: Dataset, sensitive_attribute, learning_settings):
    promotion_candidates, demotion_candidates = ranking(dataset, sensitive_attribute, learning_settings)

from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from datasets import Dataset
from helpers import set_dataset_targets
from metrics import compute_metrics_suite


def get_model_decisions(model: object, train_data: Dataset, test_data: Dataset):

    pipeline = Pipeline([
        ('normalizer', StandardScaler()),
        ('classifier', model)
    ])

    x_train = train_data.features
    y_train = train_data.targets.to_numpy().ravel()

    if train_data.instance_weights is not None:
        pipeline.fit(x_train, y_train, classifier__sample_weight=train_data.instance_weights)
    else:
        pipeline.fit(x_train, y_train)

    y_test = test_data.targets.to_numpy().ravel()
    decisions = pipeline.predict(test_data.features)
    accuracy = accuracy_score(y_test, decisions.ravel())
    print(f'Accuracy: {accuracy}')

    return decisions

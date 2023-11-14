from datasets import Dataset
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler


def train_classifier(dataset: Dataset, classifier: object, learning_settings: dict) -> float:
    pipeline = Pipeline([
        ('normalizer', StandardScaler()),
        ('clf', classifier)
    ])

    x_train, x_test, y_train, y_test = train_test_split(dataset.features, dataset.targets,
                                                        test_size=learning_settings["test_size"],
                                                        train_size=learning_settings["train_size"])

    scores = cross_validate(pipeline, x_train, y_train.values.ravel())
    return scores['test_score'].mean()


def train_all_classifiers(dataset: Dataset, learning_settings: dict) -> dict:
    results = {}

    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC(),
        "Naive Bayes": GaussianNB(),
        "Stochastic Gradient": SGDClassifier(),
        "K-Nearest Neighbours": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    for classifier in classifiers:
        accuracy = train_classifier(dataset, classifiers[classifier], learning_settings)
        results.update({classifier: accuracy})

    return results

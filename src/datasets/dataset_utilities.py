from sklearn.preprocessing import LabelEncoder

from constants import PRIVILEGED, UNPRIVILEGED


def convert_categorical_into_numerical(dataframe):
    label_encoder = LabelEncoder()
    labels_mapping = {}

    for column in dataframe.columns:
        if dataframe[column].dtype == object:
            dataframe[column] = label_encoder.fit_transform(dataframe[column])
            mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            labels_mapping.update({column: mapping})

    return dataframe, labels_mapping


def drop_invalid_instances(dataframe, indexes=None):
    if indexes is None:
        indexes = []
    if len(indexes) == 0:
        indexes = [index for index, row in dataframe.iterrows() if row.isnull().any()]
    dataframe = dataframe.drop(indexes)
    return dataframe, indexes


def is_privileged(instance: str, privileged: str):
    return PRIVILEGED if instance == privileged else UNPRIVILEGED
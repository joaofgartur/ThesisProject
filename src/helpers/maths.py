import numpy as np
import pandas as pd

from constants import NUM_DECIMALS


def ratio(a: float, b: float) -> float:
    try:
        return np.round(a / b, decimals=NUM_DECIMALS)
    except ZeroDivisionError:
        return 0.0


def diff(a: float, b: float) -> float:
    return np.round(a - b, decimals=NUM_DECIMALS)


def abs_diff(a: float, b: float) -> float:
    return np.round(abs(a - b), decimals=NUM_DECIMALS)


def joint_probability(data: pd.DataFrame, variables: dict) -> float:
    filtered_data = data.copy()
    for var, value in variables.items():
        filtered_data = filtered_data[filtered_data[var] == value]

    n_total = len(data)
    n_target = len(filtered_data)

    return ratio(n_target, n_total)


def conditional_probability(data: pd.DataFrame, variables: dict) -> float:
    denominator_variables = variables
    numerator_variables = {key: value for key, value in variables.items() if key != next(iter(variables))}

    return ratio(joint_probability(data, denominator_variables), joint_probability(data, numerator_variables))

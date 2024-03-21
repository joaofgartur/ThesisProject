"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""

import pandas as pd


def joint_probability(data: pd.DataFrame, variables: dict) -> float:

    filtered_data = data.copy()
    for var, value in variables.items():
        filtered_data = filtered_data[filtered_data[var] == value]

    n_total = len(data)
    n_target = len(filtered_data)

    try:
        return n_target / n_total
    except ZeroDivisionError:
        return 0.0

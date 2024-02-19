"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""

import pandas as pd


def safe_division(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def joint_probability(data: pd.DataFrame, variables: dict) -> float:
    n_total = len(data)

    for var, value in variables.items():
        data = data[data[var] == value]

    n_target = len(data)

    return n_target / n_total if n_total != 0 else 0.0

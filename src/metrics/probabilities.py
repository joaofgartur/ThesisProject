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


def conditional_probability(data: pd.DataFrame, a_label: str, a_value: str, b_label: str, b_value: str) -> float:
    """
    Compute the conditional probability P(A=a|B=b).

    Parameters
    ----------
    data :
        The DataFrame containing the dataset.
    a_label :
        The label of variable A.
    a_value :
        The value of variable A.
    b_label :
        The label of variable B.
    b_value :
        The value of variable B.

    Returns
    -------
    float
        The computed conditional probability.

    Notes
    -----
    - The conditional probability is calculated as the joint probability of A and B divided by the simple probability
      of B.
    - If the simple probability of B is 0, the function returns 1 to avoid division by zero.
    """
    try:
        return joint_probability(data, a_label, a_value, b_label, b_value) / simple_probability(data, b_label, b_value)
    except ZeroDivisionError:
        return 0.0



def simple_probability(data: pd.DataFrame, a_label: str, a_value: str) -> float:
    """
    Compute the simple probability P(A=a).

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the dataset.
    a_label : str
        The label of variable A.
    a_value : str
        The value of variable A.

    Returns
    -------
    float
        The computed simple probability.
    """
    n_a = data[data[a_label] == a_value].shape[0]
    n = data.shape[0]
    return n_a / n

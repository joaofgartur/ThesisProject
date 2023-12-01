"""
Author: João Artur
Project: Master's Thesis
Last edited: 30-11-2023
"""
from algorithms.DisparateImpactRemover import DisparateImpactRemover
from algorithms.LearningFairRepresentations import LearningFairRepresentations
from algorithms.OptimizedPreprocessing import OptimizedPreprocessing
from datasets import GermanCredit, AdultIncome, Compas
from algorithms import bias_correction_algorithm

import numpy as np

def get_distortion_adult2(vold, vnew):
    """Distortion function for the adult dataset. We set the distortion
    metric here. See section 4.3 in supplementary material of
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
    for an example

    Note:
        Users can use this as templates to create other distortion functions.

    Args:
        vold (dict) : {attr:value} with old values
        vnew (dict) : dictionary of the form {attr:value} with new values

    Returns:
        d (value) : distortion value
    """

    # Define local functions to adjust education and age
    def adjustEdu(v):
        if v == '>12':
            return 13
        elif v == '<6':
            return 5
        else:
            return int(v)

    def adjustAge(a):
        if a == '>=70':
            return 70.0
        else:
            return float(a)

    def adjustInc(a):
        if a == "<=50K":
            return 0
        elif a == ">50K":
            return 1
        else:
            return int(a)

    # value that will be returned for events that should not occur
    bad_val = 3.0

    # Adjust education years
    eOld = adjustEdu(vold['education'])
    eNew = adjustEdu(vnew['education'])

    # Education cannot be lowered or increased in more than 1 year
    if (eNew < eOld) | (eNew > eOld + 1):
        return bad_val

    # adjust age
    aOld = adjustAge(vold['age'])
    aNew = adjustAge(vnew['age'])

    # Age cannot be increased or decreased in more than a decade
    if np.abs(aOld - aNew) > 10.0:
        return bad_val

    # Penalty of 2 if age is decreased or increased
    if np.abs(aOld - aNew) > 0:
        return 2.0

    # Adjust income
    incOld = adjustInc(vold['income'])
    incNew = adjustInc(vnew['income'])

    # final penalty according to income
    if incOld > incNew:
        return 1.0
    else:
        return 0.0


if __name__ == '__main__':
    _learning_settings = {"train_size": 0.7, "test_size": 0.3, "seed": 125}
    optim_options = {
        "distortion_fun": get_distortion_adult2,
        "epsilon": 0.05,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
    }

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

    german_info = {
        "dataset_name": "german",
        "sensitive_attributes": {
            "Attribute9": {
                "values": ["Female", "Male"],
                "unprivileged_value": "Female",
            }
        }
    }
    german = GermanCredit(german_info)

    adult_info = {
        "dataset_name": "adult",
        "sensitive_attributes": {"race": "White", "sex": "Male"}
    }
    adult = AdultIncome(adult_info)

    algorithms = [
        DisparateImpactRemover(repair_level=1.0, learning_settings=_learning_settings),
        LearningFairRepresentations(learning_settings=_learning_settings),
        # OptimizedPreprocessing(learning_settings=_learning_settings, optimization_parameters=optim_options, features_to_keep=["age", "race", "sex", "education"]),
    ]

    for algorithm in algorithms:
        bias_correction_algorithm(adult, _learning_settings, algorithm)

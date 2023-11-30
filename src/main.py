"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""

from datasets import GermanCredit, AdultIncome, Compas
from algorithms import massaging, reweighing, bias_correction_algorithm


if __name__ == '__main__':
    _learning_settings = {"train_size": 0.7, "test_size": 0.3, "seed": 125}

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
        "sensitive_attributes": {
            "race": {
                "values": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
                "unprivileged_value": "Black",
            },
            "sex": {
                "values": ["Female", "Male"],
                "unprivileged_value": "Female",
            }
        }
    }
    adult = AdultIncome(adult_info)

    bias_correction_algorithm(compas, _learning_settings, massaging)
    bias_correction_algorithm(german, _learning_settings, massaging)
    bias_correction_algorithm(adult, _learning_settings, massaging)

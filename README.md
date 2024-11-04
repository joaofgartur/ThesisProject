# ThesisProject

## Overview

This project focuses on bias correction in datasets using various algorithms and evaluation metrics. The goal is to ensure fairness in machine learning models by addressing biases present in the data.

## Features

- **Datasets**:
  - Law School
  - Adult Income
  - German Credit
  - [**FUTURE WORK**] Other datasets might be included in the future.

- **Evaluation Metrics**:
  - **Fairness Metrics**:
    - Disparate Impact
    - Discrimination Score
    - True Positive Rate Difference
    - False Positive Rate Difference
    - Consistency
    - False Positive Error Rate Balance Score
    - False Negative Error Rate Balance Score
  - **Performance Metrics**:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC AUC
  - [**FUTURE WORK**] Other metrics might be included in the future.

- **Baselines**:
  - Massaging
  - Reweighing
  - Disparate Impact Remover
  - Lexicographic Genetic Algorithm for Fair Feature Selection
  - Multi-class Lexicographic Genetic Algorithm for Fair Feature Selection

## Future Work

- Implementation of other baseline metrics
- Implementation of visualization plots
- Extension to other relevant datasets and/or metrics
- Addition of error checking and/or handling
- Development of bias correction methods that handle multiple attributes and multi-value attributes

## Installation

Ensure you have Python installed (version 3.11) and the following libraries:

- datasets\~=2.12.0
- coloredlogs\~=15.0.1
- pandas\~=2.0.3
- numpy\~=1.24.3
- aif360\~=0.5.0
- scikit-learn\~=1.1.3
- ucimlrepo\~=0.0.3
- matplotlib\~=3.7.2

## Execution

After installing all the packages, you can execute the `run.sh` file. In this file, the supported datasets alongside the current available baseline methods are provided.

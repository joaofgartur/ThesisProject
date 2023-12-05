# ThesisProject

## Overview

## Features

- **Datasets**:
  - COMPAS
  - Adult Income
  - German Credit
  - [**FUTURE WORK**] Other datasets might be included in the future.

- **Evaluation Metrics**:
  - Discrimination Score
  - Disparate Impact
  - [**FUTURE WORK**] Other metrics might be included in the future.
- **Baselines**:
  - Massaging
  - Reweighing
  - Learning Fair Representations
  - Disparate Impact Remover
  - [**BROKEN**] Optimized Pre-processing
  - [**TODO**] No preprocessing
  - [**TODO**] Removal of the sensitive attribute

## Future work

  - Implementation of other baseline metrics
  - Implementation of visualization plots
  - Extension to other relevant dataset and/or metrics
  - Addition of error checking and/or handling
  - Development of bias correction method that handles multiple attributes and multi-value attributes

## Installation

Ensure you have Python installed (version 3.11) and the following libraries:

- datasets~=2.12.0
- coloredlogs~=15.0.1
- pandas~=2.0.3
- numpy~=1.24.3
- aif360~=0.5.0
- scikit-learn~=1.1.3
- ucimlrepo~=0.0.3
- matplotlib~=3.7.2

## Execution

- After installing all the packages, you can execute the main.py file. In this file, the supported datasets alongside 
the current available baseline methods are provided.

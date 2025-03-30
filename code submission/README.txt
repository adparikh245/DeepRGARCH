# Replication Files for "A Long Short-Term Memory Enhanced Realized Conditional Heteroskedasticity Model"

## Overview

This repository contains the code implementation for our paper, which examines the potential of using realized volatility measures for capturing financial markets' uncertainty. We conducted a comprehensive empirical study using 31 indices from 2004 to 2021. Our results demonstrate that the proposed framework achieves superior in-sample and out-of-sample performance compared to several benchmark models. Importantly, it retains interpretability and effectively adapts to the stylized facts observed in volatility, emphasizing its significant potential for enhancing economic decision-making and risk management.


## Code Contents

This repository includes implementations of the following models:
- GARCH
- RECH
- RealGARCH
- Realized RECH

Users can utilize these models to obtain preliminary results for variables of interest.

## Repository Structure

1. **src/**: This folder contains 5 files utilized in the paper for model estimation and evaluation.

2. **train.ipynb**: This primary Jupyter Notebook file is responsible for model training. It saves trained models in the checkpoint folder. Users should execute this file.

3. **experiment.ipynb**: This primary Jupyter Notebook file is responsible for model evaluation and generating results. Users should execute this file.

4. **checkpoint/**: This folder stores estimated model parameters.

5. **data/**: This folder contains the dataset used in the study.

## Usage Instructions

1. Clone this repository to your local machine.
2. Run `train.ipynb` to train the models.
3. Run `experiment.ipynb` to evaluate the models and generate results.

## Citation

Liu, C., Tran, M.-N., Wang, C., & Kohn, R. (2024). A Long Short-Term Memory Enhanced Realized Conditional Heteroskedasticity Model.

## Contact

For any questions or issues, please open an issue in this repository or contact liu.chen@sydney.edu.au.

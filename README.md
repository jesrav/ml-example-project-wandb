# MLOps batch example project, using WANDB
## Tools 
- Hydra for configuration 
- Weights and Biases for experiment tracking and artifact versioning
- Evidently AI for drift detection

# Get started

## Requirements
- Weights and Biases account, which is free for personal use: https://wandb.ai/home.
When you run one of the pipelines, you will be prompted for an API key, which can be found under settings.
- Conda

## Install
### Install dependencies
```bash
conda env create --file conda.yml 
```
### Install project as package
```bash
pip install -e .
```

## Run project
### Run training pipeline
```bash
make train_pipeline
```
### Run inference pipeline
```bash
make inference_pipeline
```
### Run drift detection on newest predictions
```bash
make drift_detection
```
### Run hyperparameter sweep with random forest model
```bash
make sweep_random_forest
```

# Scope
Not concerned with deployment. 

# TODO:
Refactor model promotion and drift detection scripts.
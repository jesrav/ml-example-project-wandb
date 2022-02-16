# MLOps batch example project, using WANDB
Toy project, that implements passive retraining for a batch prediction regression use case.

The project is not concerned with deployment or the actual ML models. It uses the Boston housing data and can be run locally.

## Tools used
- Hydra for configuration 
- Weights and Biases for experiment tracking and artifact versioning
- MLFlow for model packaging
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
This will run a training pipeline that will train a model, test it and potentially promote it to production status (by tagging the model arrtifact with a `prod` tag.
### Run inference pipeline
```bash
make inference_pipeline
```
This will run an inference pipeline that will use the `prod` model to make predictions on new data (just a sample from the Boston housing data).
A very simplistic drift can be configures in the `main` Hydra configuration.

### Run drift detection on newest predictions
```bash
make drift_detection
```
This will run drift detection, that compares the data used to make the latest predictions with the data used to train the latest `prod` model.

### Run hyperparameter sweep with random forest model
```bash
make sweep_random_forest
```
This will prepare a Weights and Biases hyperparameter sweep. You will be prompted in the terminal on how to actually run the sweep agent. 

train_pipeline: get_raw_data_train preprocess_data_train add_features_train
train_pipeline: train_validate_split cross_validation_evaluation
###############################################################
# Train pipeline
###############################################################
get_raw_data_train:
	python src/data/get_raw_training_data.py main=training-pipeline artifacts=training-pipeline

preprocess_data_train:
	python src/data/process_data.py main=training-pipeline artifacts=training-pipeline

add_features_train:
	python src/data/add_features.py main=training-pipeline artifacts=training-pipeline

data_segregation:
	python src/data/data_segregation.py

train_ridge:
	python src/models/train_and_evaluate.py model=ridge

train_random_forest:
	python src/models/train_and_evaluate.py model=random_forest

sweep_ridge:
	wandb sweep conf/wandb_sweeps/ridge.yaml

sweep_random_forest:
	wandb sweep conf/wandb_sweeps/random_forest.yaml


###############################################################
# Inference pipeline
###############################################################
inference_pipeline: get_raw_inference_data preprocess_inference_data add_features
inference_pipeline: inference

get_raw_data_inference:
	python src/data/get_raw_inference_data.py main=inference-pipeline artifacts=inference-pipeline

preprocess_data_inference:
	python src/data/process_data.py main=inference-pipeline artifacts=inference-pipeline

add_features_inference:
	python src/data/add_features.py main=inference-pipeline artifacts=inference-pipeline

batch_inference:
	python src/models/inference.py main=inference-pipeline artifacts=inference-pipeline
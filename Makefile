data: get_raw_data preprocess_data add_features train_validate_split

get_raw_data:
	python src/data/get_raw_data.py

preprocess_data:
	python src/data/process_data.py

add_features:
	python src/data/add_features.py

train_validate_split:
	python src/data/data_segregation.py

cross_validation_evaluation:
	python src/models/cross_validation_evaluation.py \
	--data-in-path data/preprocessed/train.parquet \
	--model-artifact-folder model-artifacts/cross_validation \
	--model-config src.models.model_configs.rf_model_config

hyperparam_search:
	python src/models/cross_validation_optuna_search.py \
	src.models.model_configs.hyperparam_search_configs.ridge_config \
	study.pickle \
	--n-trials 10

make hold_out_evaluation:
	python src/models/evaluate_on_hold_out.py \
	--model-in-path model-artifacts/ml_pipeline_trained_on_train.pickle

predict:
	python src/models/predict.py
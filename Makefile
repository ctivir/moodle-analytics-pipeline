# Makefile for Moodle Analytics Pipeline

.PHONY: help install test clean run-pipeline run-eda run-models

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

test: ## Run unit tests
	pytest tests/ -v

clean: ## Clean up generated files
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf report/*.png
	rm -rf report/shap/*.png
	rm -rf data/processed/session_features.csv

run-pipeline: ## Run the data processing pipeline
	python -c "
	from src.pipeline import AcademicMoodlePipeline
	pipeline = AcademicMoodlePipeline('data/raw/logs.csv', 'data/raw/grades.csv')
	df = pipeline.process()
	print(f'Processed {len(df)} student-course combinations')
	"

run-eda: ## Run exploratory data analysis
	python -c "
	import pandas as pd
	from src.eda_plots import run_eda_plots
	df = pd.read_csv('data/processed/session_features.csv')
	run_eda_plots(df)
	print('EDA plots generated')
	"

run-models: ## Run model training and evaluation
	python -c "
	import pandas as pd
	from src.regression import run_regression_models
	df = pd.read_csv('data/processed/session_features.csv')
	results = run_regression_models(df)
	print('Model training completed')
	print(results.head())
	"

setup: ## Initial project setup
	@echo "Setting up Moodle Analytics Pipeline..."
	@mkdir -p data/raw data/processed
	@touch data/raw/.gitkeep data/processed/.gitkeep
	@echo "Please place your Moodle data files in data/raw/"
	@echo "  - logs.csv: Moodle log data"
	@echo "  - grades.csv: Gradebook data"
	@echo "Then run 'make install' and 'make run-pipeline'"
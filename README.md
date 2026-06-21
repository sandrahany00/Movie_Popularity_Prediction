# Movie Popularity Predictor

Predict the popularity class of a movie from its metadata and content features.

Hosted Gradio app: [habibaseif-gradio](https://habibaseif-gradio.hf.space)

## Overview

This project builds a movie popularity classification pipeline from the training data in `train_data.csv`. The notebook explores several models and preprocessing steps, then exports the final trained artifacts used for inference and the Gradio deployment.

The repository includes the full set of saved preprocessing objects and models, so the app can reproduce the same feature engineering flow used during training.

## What the project does

- Cleans and prepares structured movie metadata.
- Encodes categorical and multi-label fields.
- Processes text-based inputs with TF-IDF and dimensionality reduction.
- Trains and evaluates multiple classifiers, including Decision Tree, XGBoost, AdaBoost, and Logistic Regression.
- Saves the final model and preprocessing artifacts for reuse in the app.

## Repository Contents

- `Movie_classification.ipynb` - main training and experimentation notebook.
- `Test_Script_class.ipynb` - notebook used for testing or inference experiments.
- `train_data.csv` - training dataset.
- `test_data_class.csv` - test dataset.
- `LogisticRegression_model.pkl`, `decisionTree_model.pkl`, `xgboost_model.pkl`, `adaboost_model.pkl` - trained model artifacts.
- `scaler.pkl`, `tfidf.pkl`, `svd.pkl`, `ohe_cat.pkl`, `ohe_lang.pkl`, `label_encoder.pkl`, `selected_features.pkl`, `median_values.json`, `num_medians.pkl`, `freq_map.pkl`, `bool_modes.pkl`, `common_languages.pkl`, `high_null_cols.pkl`, `date_params.json`, `mlb_objects.pkl` - preprocessing and feature-engineering artifacts.
- `test_predictions.csv` - example prediction output.

## Model Pipeline

The notebook shows a feature-engineering workflow that combines numerical, categorical, and text-derived movie signals. The final saved estimator is a Logistic Regression model, while other classifiers were trained for comparison during experimentation.

## Run Locally

1. Clone the repository.
2. Install the Python dependencies used in the notebook and app environment.
3. Open `Movie_classification.ipynb` or `Test_Script_class.ipynb` in Jupyter or VS Code.
4. Load the saved `.pkl` and `.json` artifacts from the project root when running inference.

If you want to reproduce the training workflow, run the notebook cells in order starting from the data-loading section.

## Deployment

The project is deployed as a Gradio app here:

[habibaseif-gradio](https://habibaseif-gradio.hf.space)

Use the hosted app to try live predictions without running the notebook locally.

## Notes

- This repository is notebook-driven, so the notebooks are the main source of truth for the training and evaluation pipeline.
- The saved artifacts in the project root are important for consistent inference and deployment.

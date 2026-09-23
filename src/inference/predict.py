import os
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from src.utils.utils import load_yaml
from src import logger


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

config = load_yaml()

RUN_ID = config["run_id"]

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if not MLFLOW_TRACKING_URI:
    raise ValueError(
        "MLFLOW_TRACKING_URI environment variable is not defined"
    )

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

logger.info("Inference device: %s", DEVICE)


# --------------------------------------------------
# Load artifacts
# --------------------------------------------------

def load_artifacts():
    logger.info(
        "Loading inference artifacts from MLflow run %s",
        RUN_ID
    )

    # Download preprocessor
    preprocessor_path = mlflow.artifacts.download_artifacts(
        artifact_uri=(
            f"runs:/{RUN_ID}/"
            "preprocessor/preprocessor.joblib"
        )
    )

    preprocessor = joblib.load(preprocessor_path)

    # Load PyTorch model
    model = mlflow.pytorch.load_model(
        f"runs:/{RUN_ID}/model",
        map_location=DEVICE
    )

    model = model.to(DEVICE)
    model.eval()

    logger.info(
        "Model and preprocessor loaded successfully"
    )

    return preprocessor, model


# Load once when application starts
PREPROCESSOR, MODEL = load_artifacts()


# --------------------------------------------------
# Prepare raw input
# --------------------------------------------------

def prepare_input(raw_data):
    """
    Accept:
        - dict
        - list of dicts
        - pandas DataFrame

    Return:
        cleaned pandas DataFrame
    """

    if isinstance(raw_data, dict):
        df = pd.DataFrame([raw_data])

    elif isinstance(raw_data, list):
        df = pd.DataFrame(raw_data)

    elif isinstance(raw_data, pd.DataFrame):
        df = raw_data.copy()

    else:
        raise TypeError(
            "raw_data must be a dict, list of dicts, "
            "or pandas DataFrame"
        )

    # Inference should normally not contain target.
    # This just makes notebook/testing usage safer.
    df = df.drop(
        columns=["target"],
        errors="ignore"
    )

    # ----------------------------------------------
    # Basic cleaning used during training
    # ----------------------------------------------

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    for col in categorical_features:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .replace("?", np.nan)
            )

    # ----------------------------------------------
    # Validate expected columns
    # ----------------------------------------------

    if hasattr(PREPROCESSOR, "feature_names_in_"):

        expected_columns = list(
            PREPROCESSOR.feature_names_in_
        )

        missing_columns = [
            col
            for col in expected_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing input columns: {missing_columns}"
            )

        # Same column order used during training
        df = df[expected_columns]

    return df


# --------------------------------------------------
# Inference
# --------------------------------------------------

def predict(raw_data):

    df = prepare_input(raw_data)

    logger.info(
        "Running inference for %d sample(s)",
        len(df)
    )

    # sklearn preprocessing
    X_processed = PREPROCESSOR.transform(df)

    # Works whether sklearn returns DataFrame or ndarray
    X_array = np.asarray(
        X_processed,
        dtype=np.float32
    )

    X_tensor = torch.tensor(
        X_array,
        dtype=torch.float32,
        device=DEVICE
    )

    # PyTorch inference
    with torch.inference_mode():

        logits = MODEL(X_tensor)

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= 0.5
        ).int()

    # Convert tensors -> normal Python values
    probabilities = (
        probabilities
        .cpu()
        .numpy()
        .flatten()
        .tolist()
    )

    predictions = (
        predictions
        .cpu()
        .numpy()
        .flatten()
        .tolist()
    )

    logger.info(
        "Inference completed successfully"
    )

    return {
        "probabilities": probabilities,
        "predictions": predictions
    }
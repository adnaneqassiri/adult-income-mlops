from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.models.model import IncomeClassifier
from src.models.engine import predict
from src.utils.utils import load_yaml
from src import logger


config = load_yaml()

processed_path = Path(
    config["paths"]["processed"]
)

model_path = Path(
    config["paths"]["model"]
)


# -------------------
# Load test data
# -------------------

logger.info("Loading test dataset")

test_df = pd.read_csv(
    processed_path / "test.csv"
)

X_test = test_df.drop(
    columns="target"
)

y_test = test_df["target"]


X_test_tensor = torch.tensor(
    X_test.values,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
).unsqueeze(1)


test_loader = DataLoader(
    TensorDataset(
        X_test_tensor,
        y_test_tensor
    ),
    batch_size=128,
    shuffle=False
)


# -------------------
# Device
# -------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

logger.info(
    "Using device: %s",
    device
)


# -------------------
# Load model
# -------------------

checkpoint = torch.load(
    model_path,
    map_location=device
)

model = IncomeClassifier(
    input_dim=checkpoint["input_dim"]
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# -------------------
# Prediction
# -------------------

logger.info("Running test inference")

predictions, probabilities = predict(
    model,
    test_loader,
    device
)


# -------------------
# Metrics
# -------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)


logger.info(
    "Test metrics | Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
    accuracy,
    precision,
    recall,
    f1
)
from pathlib import Path
import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from src.models.model import IncomeClassifier
from src.models.engine import fit
from src.utils.utils import load_yaml
from src import logger

import os
from dotenv import load_dotenv

load_dotenv()

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

config = load_yaml()
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("adult-income-dnn")


processed_path = Path(config["paths"]["processed"])
model_path = Path(config["paths"]["model"])
model_path.parent.mkdir(parents=True, exist_ok=True)
preprocessor_path = Path(
    config["paths"]["preprocessor"]
)

# -------------------
# Load data
# -------------------

logger.info("Loading processed training data")

train_df = pd.read_csv(processed_path / "train.csv")
val_df = pd.read_csv(processed_path / "val.csv")

X_train = train_df.drop(columns="target")
y_train = train_df["target"]

X_val = val_df.drop(columns="target")
y_val = val_df["target"]


# -------------------
# Convert to tensors
# -------------------

X_train = torch.tensor(
    X_train.values,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train.values,
    dtype=torch.float32
).unsqueeze(1)


X_val = torch.tensor(
    X_val.values,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val.values,
    dtype=torch.float32
).unsqueeze(1)


# -------------------
# DataLoaders
# -------------------

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=128,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=128,
    shuffle=False
)


# -------------------
# Model
# -------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

logger.info("Using device: %s", device)


model = IncomeClassifier(
    input_dim=X_train.shape[1]
).to(device)


criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -------------------
# Train
# -------------------


with mlflow.start_run():

    mlflow.log_param("batch_size", 128)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", 10)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("loss", "BCEWithLogitsLoss")
    mlflow.log_param("input_dim", X_train.shape[1])

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=10,
        checkpoint_path=model_path
    )

    
    mlflow.log_artifact(
        str(preprocessor_path),
        artifact_path="preprocessor"
    )
    
    mlflow.pytorch.log_model(
        model,
        artifact_path="model"
    )
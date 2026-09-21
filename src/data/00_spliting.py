from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from src.utils.utils import load_yaml
from src import logger
from pathlib import Path


def encode_target(y):
    y = (
        y.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
    )

    y = y.map({
        "<=50K": 0,
        ">50K": 1
    })

    if y.isna().any():
        raise ValueError("Unknown target label detected")

    return y


config = load_yaml()
logger.info("Fetching Adult dataset from UCI")

adult = fetch_ucirepo(id=2)

X = adult.data.features
y = adult.data.targets.squeeze()

logger.info(
    "Dataset loaded - X shape: %s, y shape: %s",
    X.shape,
    y.shape
)

y = encode_target(y)
# 1. Train = 70%, remaining = 30%
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 2. Split remaining 30% equally -> 15% validation, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

logger.info(
    "Data split completed - Train: %s, Validation: %s, Test: %s",
    X_train.shape,
    X_val.shape,
    X_test.shape
)

X_train["target"] = y_train
X_val["target"] = y_val
X_test["target"] = y_test

raw_path = Path(config["paths"]["raw"])

X_train.to_csv(raw_path / "train.csv", index=False)
X_val.to_csv(raw_path / "val.csv", index=False)
X_test.to_csv(raw_path / "test.csv", index=False)

logger.info("Dataset splits saved successfully to %s", raw_path)
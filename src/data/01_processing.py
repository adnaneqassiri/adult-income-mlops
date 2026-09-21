from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd
import joblib

from src.utils.utils import load_yaml
from pathlib import Path
from src import logger


config = load_yaml()

raw_path = Path(config["paths"]["raw"])
processed_path = Path(config["paths"]["processed"])

target_col = "target"


# -----------------------
# Load data
# -----------------------

logger.info("Loading raw datasets")

train_df = pd.read_csv(raw_path / "train.csv")
val_df = pd.read_csv(raw_path / "val.csv")
test_df = pd.read_csv(raw_path / "test.csv")

logger.info(
    "Datasets loaded - Train: %s, Val: %s, Test: %s",
    train_df.shape,
    val_df.shape,
    test_df.shape
)


# -----------------------
# Separate features / target
# -----------------------

logger.info("Separating features and target")

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_val = val_df.drop(columns=[target_col])
y_val = val_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]


# -----------------------
# Define features
# -----------------------

numeric_features = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

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


# -----------------------
# Basic cleaning
# -----------------------

logger.info("Cleaning categorical features")

for df in [X_train, X_val, X_test]:
    df[categorical_features] = (
        df[categorical_features]
        .apply(lambda col: col.str.strip())
        .replace("?", np.nan)
    )

logger.info("Categorical cleaning completed")


# -----------------------
# Preprocessing pipelines
# -----------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
).set_output(transform="pandas")


# -----------------------
# Fit + Transform
# -----------------------

logger.info("Fitting preprocessor on training data")

X_train_processed = preprocessor.fit_transform(X_train)

logger.info("Transforming validation and test data")

X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

logger.info(
    "Preprocessing completed - Train: %s, Val: %s, Test: %s",
    X_train_processed.shape,
    X_val_processed.shape,
    X_test_processed.shape
)


# -----------------------
# Combine features + target
# -----------------------

train_processed = X_train_processed.copy()
val_processed = X_val_processed.copy()
test_processed = X_test_processed.copy()

train_processed[target_col] = y_train.to_numpy()
val_processed[target_col] = y_val.to_numpy()
test_processed[target_col] = y_test.to_numpy()


# -----------------------
# Save processed data
# -----------------------

processed_path.mkdir(parents=True, exist_ok=True)

logger.info("Saving processed datasets to %s", processed_path)

train_processed.to_csv(
    processed_path / "train.csv",
    index=False
)

val_processed.to_csv(
    processed_path / "val.csv",
    index=False
)

test_processed.to_csv(
    processed_path / "test.csv",
    index=False
)

logger.info("Processed datasets saved successfully")


# -----------------------
# Save preprocessor
# -----------------------

preprocessor_path = Path(config["paths"]["preprocessor"])

preprocessor_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    preprocessor,
    preprocessor_path
)

logger.info(
    "Preprocessor saved to %s",
    preprocessor_path
)
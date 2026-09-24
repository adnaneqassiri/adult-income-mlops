# Adult Income MLOps

An end-to-end machine learning project that predicts whether a person's annual income exceeds **$50K** using the [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult).

The repository includes data preparation, a PyTorch binary classifier, MLflow experiment and artifact tracking, a FastAPI inference service, Docker packaging, tests, and a GitHub Actions deployment pipeline.

## Architecture

```text
UCI Adult dataset
       |
       v
Train / validation / test split
       |
       v
Imputation + scaling + one-hot encoding
       |
       v
PyTorch training -----> MLflow tracking and artifacts
                              |
                              v
                         FastAPI service
                              |
                              v
                         Docker / EC2
```

## Project structure

```text
.
├── src/
│   ├── api/             # FastAPI app and request/response schemas
│   ├── data/            # Dataset download, splitting, and preprocessing
│   ├── inference/       # MLflow artifact loading and prediction
│   ├── models/          # PyTorch model, training loop, and evaluation
│   ├── utils/           # Shared utilities
│   └── config.yaml      # Data, artifact, and MLflow run configuration
├── tests/               # API, model, and schema tests
├── data/                # Raw and processed datasets
├── artifacts/           # Local model checkpoint and preprocessor
├── EDA.ipynb            # Exploratory data analysis
├── Dockerfile
└── pyproject.toml
```

## Prerequisites

- Python 3.10 or newer
- An MLflow tracking server accessible from the training and API environments
- Docker (optional)

## Installation

Using `uv`:

```bash
uv sync --extra dev
source .venv/bin/activate
```

Or using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Create a local `.env` file with the MLflow server address:

```dotenv
MLFLOW_TRACKING_URI=http://localhost:5000
```

Do not commit `.env` or cloud credentials.

## Run the ML pipeline

Run all commands from the repository root.

1. Download and split the dataset into 70% training, 15% validation, and 15% test sets:

   ```bash
   python src/data/00_spliting.py
   ```

2. Fit the preprocessing pipeline and transform each split:

   ```bash
   python src/data/01_processing.py
   ```

3. Train the neural network and log its parameters, preprocessor, and model to MLflow:

   ```bash
   python -m src.models.train
   ```

4. Copy the resulting MLflow run ID into `run_id` in `src/config.yaml`. The API uses this ID to retrieve the trained model and preprocessor.

5. Evaluate the local best checkpoint on the test split:

   ```bash
   python -m src.models.evaluate
   ```

Training uses a feed-forward PyTorch classifier, binary cross-entropy loss, and Adam. Evaluation reports accuracy, precision, recall, and F1 score in the application log.

## Run the API

Start the development server after setting `MLFLOW_TRACKING_URI` and configuring a valid MLflow `run_id`:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Useful endpoints:

- `GET /health` — service health check
- `POST /predict` — income prediction
- `GET /docs` — interactive OpenAPI documentation

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 39,
    "workclass": "Private",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
  }'
```

Example response:

```json
{
  "prediction": 0,
  "probability": 0.1234
}
```

`prediction` is `1` for income above $50K and `0` otherwise. The model and preprocessor are downloaded from MLflow on the first prediction and then cached in memory.

## Tests

```bash
pytest
```

The test suite checks the health endpoint, request validation, and model output shape.

## Docker

Build and run the API locally:

```bash
docker build -t income-api .
docker run --rm \
  --name income-api \
  --env-file .env \
  -p 8000:8000 \
  income-api
```

The container must be able to reach the configured MLflow server and its artifact store.

## CI/CD

The workflow in `.github/workflows/CI-CD.yml` runs on pushes and pull requests targeting `master`:

1. Install the package and run the tests.
2. On a push, build the image and publish `latest` and commit-SHA tags to Docker Hub.
3. Connect to EC2, replace the running container, and verify `/health`.

Configure these GitHub Actions secrets:

| Secret | Purpose |
| --- | --- |
| `DOCKERHUB_USERNAME` | Docker Hub account and image namespace |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `EC2_HOST` | EC2 hostname or public IP |
| `EC2_USER` | SSH user for the instance |
| `EC2_SSH_KEY_B64` | Base64-encoded private SSH key |

The EC2 host must have Docker installed and an environment file at `/home/ubuntu/income-api.env` containing `MLFLOW_TRACKING_URI` and any credentials required to access the MLflow artifact store.

## Configuration

Paths and the inference run ID are defined in `src/config.yaml`. Update that file when changing data locations, local artifact locations, or the production MLflow run used by the API.

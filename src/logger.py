import logging
from pathlib import Path

from src.utils.utils import load_yaml


config = load_yaml()

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Path from config.yaml
LOG_FILE = Path(config["paths"]["logs"])

# If config contains "./logs/logs.log",
# resolve it relative to the project root
if not LOG_FILE.is_absolute():
    LOG_FILE = BASE_DIR / LOG_FILE

# Create logs/ if it doesn't exist
LOG_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


logger = logging.getLogger("ml_project")
logger.setLevel(logging.INFO)

# Prevent messages from also propagating to the root logger
logger.propagate = False


if not logger.handlers:

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # File
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
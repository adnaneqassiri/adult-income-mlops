import yaml
from pathlib import Path

def load_yaml():
    # Load the configuration file
    with open(Path(__file__).resolve().parents[1] / "config.yaml", "r") as file:
        config = yaml.safe_load(file)

    return config

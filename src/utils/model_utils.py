import os

import joblib
import torch
import yaml

from src.components.gnn_model import EGFRGraphNet
from src.logger import logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def load_classical_model(path: str):
    """Load a scikit-learn / XGBoost model from disk."""
    model = joblib.load(path)
    logger.info(f"Loaded model from {path}")
    return model


def load_gnn_model(
    model_path: str,
    num_node_features: int,
    config: dict,
    device: torch.device | None = None,
) -> EGFRGraphNet:
    """Load a trained GNN model from disk."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gnn_cfg = config["model"]["gnn"]
    model = EGFRGraphNet(
        num_node_features=num_node_features,
        hidden_channels=gnn_cfg["hidden_channels"][0],
        num_layers=gnn_cfg["num_layers"][0],
        dropout=gnn_cfg["dropout"][0],
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    logger.info(f"Loaded GNN model from {model_path}")
    return model

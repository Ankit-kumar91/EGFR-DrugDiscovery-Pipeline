import os

import numpy as np
import pandas as pd
import torch

from src.components.feature_engineering import MolecularFeatureEngineer
from src.components.virtual_screening import VirtualScreener
from src.components.model_monitoring import ModelMonitor
from src.logger import logger
from src.utils.model_utils import load_config, load_classical_model, load_gnn_model


class PredictPipeline:
    """Prediction pipeline for single compounds and virtual screening."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.engineer = MolecularFeatureEngineer(self.config)
        self.monitor = ModelMonitor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-loaded models
        self._rf_model = None
        self._xgb_model = None
        self._gnn_model = None

    @property
    def rf_model(self):
        if self._rf_model is None:
            self._rf_model = load_classical_model("models/random_forest_best.pkl")
        return self._rf_model

    @property
    def xgb_model(self):
        if self._xgb_model is None:
            self._xgb_model = load_classical_model("models/xgboost_best.pkl")
        return self._xgb_model

    @property
    def gnn_model(self):
        if self._gnn_model is None:
            graphs = torch.load("data/processed/egfr_graphs.pt", weights_only=False)
            num_node_features = graphs[0].x.shape[1]
            self._gnn_model = load_gnn_model(
                "models/gnn_best.pt", num_node_features, self.config, self.device
            )
        return self._gnn_model

    def predict_single(self, smiles: str) -> dict:
        """Predict pIC50 for a single SMILES string."""
        logger.info(f"Predicting pIC50 for: {smiles}")

        # Featurize
        fp = self.engineer.compute_fingerprints([smiles])
        desc = self.engineer.compute_descriptors([smiles])
        X = np.hstack([fp, desc.values])

        # Classical ML predictions
        rf_pred = float(self.rf_model.predict(X)[0])
        xgb_pred = float(self.xgb_model.predict(X)[0])

        # GNN prediction
        graph = self.engineer.smiles_to_graph(smiles)
        if graph is not None:
            from torch_geometric.loader import DataLoader
            loader = DataLoader([graph], batch_size=1)
            self.gnn_model.eval()
            with torch.no_grad():
                batch = next(iter(loader)).to(self.device)
                gnn_pred = float(self.gnn_model(batch).cpu().item())
        else:
            gnn_pred = None

        # Consensus
        preds = [rf_pred, xgb_pred]
        if gnn_pred is not None:
            preds.append(gnn_pred)
        consensus = float(np.mean(preds))

        # Lipinski properties
        lipinski = self.engineer.compute_lipinski([smiles]).iloc[0].to_dict()

        result = {
            "smiles": smiles,
            "pIC50_RF": round(rf_pred, 4),
            "pIC50_XGB": round(xgb_pred, 4),
            "pIC50_GNN": round(gnn_pred, 4) if gnn_pred is not None else None,
            "pIC50_consensus": round(consensus, 4),
            "activity": "active" if consensus >= 6.0 else "inactive",
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in lipinski.items()},
        }

        # Log prediction
        self.monitor.log_prediction(smiles, consensus, "consensus")

        return result

    def run_virtual_screening(
        self, library_path: str = "data/libraries/enamine_diversity.csv"
    ) -> pd.DataFrame:
        """Run virtual screening on a compound library."""
        logger.info("Starting virtual screening")
        screener = VirtualScreener(self.config)
        hits = screener.run(
            library_path=library_path,
            rf_model=self.rf_model,
            xgb_model=self.xgb_model,
            gnn_model=self.gnn_model,
            output_path="results/virtual_screening_hits.csv",
        )
        return hits

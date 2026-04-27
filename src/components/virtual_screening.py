import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.components.feature_engineering import MolecularFeatureEngineer
from src.components.gnn_model import EGFRGraphNet
from src.logger import logger
from src.exception import PipelineException


class VirtualScreener:
    """Screen compound libraries using trained pIC50 prediction models."""

    def __init__(self, config: dict):
        self.config = config
        vs_cfg = config["virtual_screening"]
        self.pIC50_threshold = vs_cfg["pIC50_threshold"]
        self.top_n_hits = vs_cfg["top_n_hits"]
        self.feature_engineer = MolecularFeatureEngineer(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_enamine_library(self, path: str) -> pd.DataFrame:
        """Load a compound library (Enamine Diversity Set or similar) from CSV/SDF."""
        logger.info(f"Loading compound library from {path}")
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".smi"):
            df = pd.read_csv(path, sep="\t", header=None, names=["SMILES", "Name"])
            df.rename(columns={"SMILES": "canonical_smiles", "Name": "compound_id"}, inplace=True)
        else:
            raise PipelineException(f"Unsupported file format: {path}", sys)
        logger.info(f"Loaded {len(df)} compounds from library")
        return df

    def featurize_library(self, df: pd.DataFrame) -> tuple[np.ndarray, list]:
        """Generate fingerprints and graph representations for the library."""
        smiles_list = df["canonical_smiles"].tolist()
        logger.info(f"Featurizing {len(smiles_list)} compounds")

        # Fingerprints + descriptors for classical ML
        fp_array = self.feature_engineer.compute_fingerprints(smiles_list)
        desc_df = self.feature_engineer.compute_descriptors(smiles_list)
        X = np.hstack([fp_array, desc_df.values])

        # Graphs for GNN
        graph_data = self.feature_engineer.build_graph_dataset(smiles_list)

        return X, graph_data

    def screen_with_classical(
        self, model, X: np.ndarray, model_name: str
    ) -> np.ndarray:
        """Predict pIC50 using a classical ML model (RF or XGBoost)."""
        logger.info(f"Screening with {model_name}")
        predictions = model.predict(X)
        return predictions

    def screen_with_gnn(
        self, model: EGFRGraphNet, graph_data: list
    ) -> np.ndarray:
        """Predict pIC50 using the GNN model."""
        logger.info("Screening with GNN")
        model.eval()
        loader = DataLoader(graph_data, batch_size=64, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                pred = model(batch)
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds)

    def consensus_scoring(self, predictions_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Compute consensus score by averaging predictions across models."""
        logger.info(f"Computing consensus score from {len(predictions_dict)} models")
        all_preds = np.stack(list(predictions_dict.values()), axis=0)
        consensus = np.mean(all_preds, axis=0)
        return consensus

    def filter_hits(
        self, df: pd.DataFrame, predictions: np.ndarray, threshold: float | None = None
    ) -> pd.DataFrame:
        """Apply pIC50 cutoff and Lipinski filter to identify hits."""
        threshold = threshold or self.pIC50_threshold
        df = df.copy()
        df["predicted_pIC50"] = predictions

        # pIC50 threshold filter
        hits = df[df["predicted_pIC50"] >= threshold].copy()
        logger.info(f"Compounds with predicted pIC50 >= {threshold}: {len(hits)}")

        # Lipinski filter
        lipinski_df = self.feature_engineer.compute_lipinski(
            hits["canonical_smiles"].tolist()
        )
        hits = pd.concat([hits.reset_index(drop=True), lipinski_df], axis=1)
        hits = hits[hits["Lipinski_Pass"]].copy()
        logger.info(f"After Lipinski filter: {len(hits)} hits")

        # Sort by predicted pIC50
        hits = hits.sort_values("predicted_pIC50", ascending=False).reset_index(drop=True)

        # Take top N
        hits = hits.head(self.top_n_hits)
        logger.info(f"Top {len(hits)} hits selected")
        return hits

    def export_hits(self, hits_df: pd.DataFrame, output_path: str):
        """Save hit compounds to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        hits_df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(hits_df)} hits to {output_path}")

    def run(
        self,
        library_path: str,
        rf_model=None,
        xgb_model=None,
        gnn_model: EGFRGraphNet | None = None,
        output_path: str = "results/virtual_screening_hits.csv",
    ) -> pd.DataFrame:
        """Execute the full virtual screening pipeline."""
        logger.info("Starting virtual screening pipeline")
        library_df = self.load_enamine_library(library_path)
        X, graph_data = self.featurize_library(library_df)

        predictions = {}
        if rf_model is not None:
            predictions["RF"] = self.screen_with_classical(rf_model, X, "RandomForest")
        if xgb_model is not None:
            predictions["XGB"] = self.screen_with_classical(xgb_model, X, "XGBoost")
        if gnn_model is not None:
            # GNN may have fewer valid molecules
            gnn_preds = self.screen_with_gnn(gnn_model, graph_data)
            # Pad to match library length if needed
            if len(gnn_preds) < len(library_df):
                padded = np.full(len(library_df), np.nan)
                padded[: len(gnn_preds)] = gnn_preds
                predictions["GNN"] = padded
            else:
                predictions["GNN"] = gnn_preds

        if not predictions:
            raise PipelineException("No models provided for screening", sys)

        consensus = self.consensus_scoring(predictions)

        # Add individual model predictions to the dataframe
        for name, preds in predictions.items():
            library_df[f"pIC50_{name}"] = preds

        hits = self.filter_hits(library_df, consensus)
        self.export_hits(hits, output_path)
        logger.info("Virtual screening pipeline complete")
        return hits

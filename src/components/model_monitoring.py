import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from src.logger import logger


class ModelMonitor:
    """Monitor model predictions and detect data drift."""

    def __init__(self, log_dir: str = "results/monitoring"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.prediction_log_path = os.path.join(log_dir, "prediction_log.jsonl")

    def detect_data_drift(
        self,
        reference_data: np.ndarray,
        new_data: np.ndarray,
        feature_names: list[str] | None = None,
        significance_level: float = 0.05,
    ) -> dict:
        """Detect data drift using the Kolmogorov-Smirnov test.

        Compares the distribution of each feature between reference and new data.
        Returns a dict with per-feature p-values and drift flags.
        """
        logger.info("Running data drift detection")
        if reference_data.shape[1] != new_data.shape[1]:
            raise ValueError(
                f"Feature dimensions do not match: "
                f"reference={reference_data.shape[1]}, new={new_data.shape[1]}"
            )

        n_features = reference_data.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        results = {}
        drifted_features = []

        for i, name in enumerate(feature_names):
            stat, p_value = stats.ks_2samp(reference_data[:, i], new_data[:, i])
            is_drifted = p_value < significance_level
            results[name] = {
                "ks_statistic": round(stat, 4),
                "p_value": round(p_value, 6),
                "drift_detected": is_drifted,
            }
            if is_drifted:
                drifted_features.append(name)

        drift_report = {
            "timestamp": datetime.now().isoformat(),
            "n_features": n_features,
            "n_drifted": len(drifted_features),
            "drift_fraction": round(len(drifted_features) / n_features, 4),
            "drifted_features": drifted_features,
            "details": results,
        }

        logger.info(
            f"Drift detection: {len(drifted_features)}/{n_features} features drifted "
            f"({drift_report['drift_fraction']:.1%})"
        )

        # Save report
        report_path = os.path.join(self.log_dir, "drift_report.json")
        with open(report_path, "w") as f:
            json.dump(drift_report, f, indent=2)
        logger.info(f"Drift report saved to {report_path}")

        return drift_report

    def log_prediction(
        self,
        input_smiles: str,
        predicted_pIC50: float,
        model_name: str = "unknown",
    ):
        """Log a single prediction to the prediction log."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "smiles": input_smiles,
            "predicted_pIC50": round(predicted_pIC50, 4),
            "model": model_name,
        }
        with open(self.prediction_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def generate_monitoring_report(self) -> dict:
        """Generate a summary report from the prediction log."""
        logger.info("Generating monitoring report")

        if not os.path.exists(self.prediction_log_path):
            logger.warning("No prediction log found")
            return {"total_predictions": 0}

        records = []
        with open(self.prediction_log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            return {"total_predictions": 0}

        df = pd.DataFrame(records)
        pIC50_values = df["predicted_pIC50"]

        report = {
            "total_predictions": len(df),
            "unique_smiles": df["smiles"].nunique(),
            "models_used": df["model"].unique().tolist(),
            "pIC50_stats": {
                "mean": round(pIC50_values.mean(), 4),
                "std": round(pIC50_values.std(), 4),
                "min": round(pIC50_values.min(), 4),
                "max": round(pIC50_values.max(), 4),
            },
            "active_predictions": int((pIC50_values >= 6.0).sum()),
            "inactive_predictions": int((pIC50_values < 6.0).sum()),
            "time_range": {
                "first": df["timestamp"].min(),
                "last": df["timestamp"].max(),
            },
        }

        report_path = os.path.join(self.log_dir, "monitoring_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Monitoring report saved to {report_path}")
        return report

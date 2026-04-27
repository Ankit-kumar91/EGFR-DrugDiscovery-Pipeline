import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from torch_geometric.loader import DataLoader

from src.logger import logger


class ModelEvaluator:
    """Evaluate and compare pIC50 prediction models."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute regression metrics."""
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
        return {
            "R2": round(r2, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "Pearson_r": round(pearson_r, 4),
            "Pearson_p": round(pearson_p, 6),
        }

    def predict_gnn(self, model, dataset: list, device: torch.device) -> np.ndarray:
        """Get predictions from a GNN model."""
        model.eval()
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = model(batch)
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds)

    def plot_actual_vs_predicted(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str
    ):
        """Scatter plot of actual vs predicted pIC50."""
        metrics = self.evaluate_regression(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="steelblue")

        # Perfect prediction line
        lims = [min(y_true.min(), y_pred.min()) - 0.5, max(y_true.max(), y_pred.max()) + 0.5]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal")

        ax.set_xlabel("Actual pIC50")
        ax.set_ylabel("Predicted pIC50")
        ax.set_title(f"{model_name}: Actual vs Predicted")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.legend()

        textstr = f"R² = {metrics['R2']:.3f}\nRMSE = {metrics['RMSE']:.3f}\nMAE = {metrics['MAE']:.3f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        path = os.path.join(self.plots_dir, f"{model_name}_actual_vs_predicted.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved actual vs predicted plot to {path}")

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Residual plot."""
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(y_pred, residuals, alpha=0.4, s=15, color="steelblue")
        axes[0].axhline(y=0, color="red", linestyle="--")
        axes[0].set_xlabel("Predicted pIC50")
        axes[0].set_ylabel("Residual")
        axes[0].set_title(f"{model_name}: Residuals vs Predicted")

        sns.histplot(residuals, bins=50, kde=True, ax=axes[1], color="steelblue")
        axes[1].set_xlabel("Residual")
        axes[1].set_title(f"{model_name}: Residual Distribution")

        path = os.path.join(self.plots_dir, f"{model_name}_residuals.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved residual plot to {path}")

    def plot_feature_importance(
        self, model, feature_names: list[str], model_name: str, top_n: int = 20
    ):
        """Plot top N feature importances for tree-based models."""
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]

        fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
        ax.barh(
            [feature_names[i] for i in indices],
            importances[indices],
            color="steelblue",
        )
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"{model_name}: Top {top_n} Feature Importances")

        path = os.path.join(self.plots_dir, f"{model_name}_feature_importance.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved feature importance plot to {path}")

    def compare_models(self, results_dict: dict[str, dict]) -> pd.DataFrame:
        """Create a comparison table from results dict.

        Args:
            results_dict: {model_name: {"y_true": ..., "y_pred": ...}}
        """
        rows = []
        for model_name, data in results_dict.items():
            metrics = self.evaluate_regression(data["y_true"], data["y_pred"])
            metrics["Model"] = model_name
            rows.append(metrics)
        df = pd.DataFrame(rows).set_index("Model")
        path = os.path.join(self.output_dir, "model_comparison.csv")
        df.to_csv(path)
        logger.info(f"Saved model comparison to {path}")
        return df

    def plot_learning_curves(
        self, model, X, y, model_name: str, cv: int = 5
    ):
        """Plot learning curves for a scikit-learn compatible model."""
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        train_rmse = -train_scores.mean(axis=1)
        val_rmse = -val_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_std = val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes, train_rmse, "o-", label="Training RMSE", color="steelblue")
        ax.fill_between(train_sizes, train_rmse - train_std, train_rmse + train_std, alpha=0.1, color="steelblue")
        ax.plot(train_sizes, val_rmse, "o-", label="Validation RMSE", color="darkorange")
        ax.fill_between(train_sizes, val_rmse - val_std, val_rmse + val_std, alpha=0.1, color="darkorange")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("RMSE")
        ax.set_title(f"{model_name}: Learning Curves")
        ax.legend()

        path = os.path.join(self.plots_dir, f"{model_name}_learning_curves.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved learning curves to {path}")

    def generate_report(self, results_dict: dict[str, dict]) -> str:
        """Generate all evaluation plots and comparison table."""
        logger.info("Generating evaluation report")
        for model_name, data in results_dict.items():
            self.plot_actual_vs_predicted(data["y_true"], data["y_pred"], model_name)
            self.plot_residuals(data["y_true"], data["y_pred"], model_name)

        comparison_df = self.compare_models(results_dict)
        logger.info(f"\n{comparison_df.to_string()}")
        return os.path.join(self.output_dir, "model_comparison.csv")

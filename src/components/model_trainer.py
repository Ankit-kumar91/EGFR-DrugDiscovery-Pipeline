import os
import sys
import copy

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from torch_geometric.loader import DataLoader
from xgboost import XGBRegressor

from src.components.gnn_model import EGFRGraphNet
from src.logger import logger
from src.exception import PipelineException


class IC50ModelTrainer:
    """Train RF, XGBoost, and GNN models for pIC50 prediction with MLflow tracking."""

    def __init__(self, config: dict):
        self.config = config
        model_cfg = config["model"]
        self.test_size = model_cfg["test_size"]
        self.random_state = model_cfg["random_state"]
        self.cv_folds = model_cfg["cv_folds"]
        self.rf_params = model_cfg["random_forest"]
        self.xgb_params = model_cfg["xgboost"]
        self.gnn_params = model_cfg["gnn"]

        mlflow_cfg = config["mlflow"]
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        mlflow.set_experiment(mlflow_cfg["experiment_name"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs("models", exist_ok=True)

    # ── Classical ML ──────────────────────────────────────────────────

    def train_random_forest(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> RandomForestRegressor:
        """Train Random Forest with RandomizedSearchCV and log to MLflow."""
        logger.info("Training Random Forest with hyperparameter search")
        param_dist = {
            "n_estimators": self.rf_params["n_estimators"],
            "max_depth": self.rf_params["max_depth"],
            "min_samples_split": self.rf_params["min_samples_split"],
        }

        with mlflow.start_run(run_name="RandomForest") as run:
            search = RandomizedSearchCV(
                RandomForestRegressor(random_state=self.random_state),
                param_distributions=param_dist,
                n_iter=10,
                cv=self.cv_folds,
                scoring="neg_root_mean_squared_error",
                random_state=self.random_state,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            # Log to MLflow
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("best_cv_rmse", -search.best_score_)

            cv_r2 = cross_val_score(
                best_model, X_train, y_train, cv=self.cv_folds, scoring="r2"
            ).mean()
            mlflow.log_metric("cv_r2", cv_r2)
            mlflow.sklearn.log_model(best_model, "random_forest")

            logger.info(f"RF best params: {search.best_params_}")
            logger.info(f"RF CV RMSE: {-search.best_score_:.4f}, CV R²: {cv_r2:.4f}")

        path = "models/random_forest_best.pkl"
        joblib.dump(best_model, path)
        logger.info(f"Saved RF model to {path}")
        return best_model

    def train_xgboost(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> XGBRegressor:
        """Train XGBoost with RandomizedSearchCV and log to MLflow."""
        logger.info("Training XGBoost with hyperparameter search")
        param_dist = {
            "n_estimators": self.xgb_params["n_estimators"],
            "max_depth": self.xgb_params["max_depth"],
            "learning_rate": self.xgb_params["learning_rate"],
            "subsample": self.xgb_params["subsample"],
        }

        with mlflow.start_run(run_name="XGBoost") as run:
            search = RandomizedSearchCV(
                XGBRegressor(random_state=self.random_state, verbosity=0),
                param_distributions=param_dist,
                n_iter=10,
                cv=self.cv_folds,
                scoring="neg_root_mean_squared_error",
                random_state=self.random_state,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            mlflow.log_params(search.best_params_)
            mlflow.log_metric("best_cv_rmse", -search.best_score_)

            cv_r2 = cross_val_score(
                best_model, X_train, y_train, cv=self.cv_folds, scoring="r2"
            ).mean()
            mlflow.log_metric("cv_r2", cv_r2)
            mlflow.sklearn.log_model(best_model, "xgboost")

            logger.info(f"XGB best params: {search.best_params_}")
            logger.info(f"XGB CV RMSE: {-search.best_score_:.4f}, CV R²: {cv_r2:.4f}")

        path = "models/xgboost_best.pkl"
        joblib.dump(best_model, path)
        logger.info(f"Saved XGBoost model to {path}")
        return best_model

    # ── GNN ───────────────────────────────────────────────────────────

    def build_gnn_model(self, num_node_features: int) -> EGFRGraphNet:
        """Instantiate the GNN model."""
        model = EGFRGraphNet(
            num_node_features=num_node_features,
            hidden_channels=self.gnn_params["hidden_channels"][0],
            num_layers=self.gnn_params["num_layers"][0],
            dropout=self.gnn_params["dropout"][0],
        )
        return model.to(self.device)

    def train_gnn(
        self,
        train_dataset: list,
        val_dataset: list,
        num_node_features: int,
    ) -> EGFRGraphNet:
        """Train GNN with early stopping and log to MLflow."""
        logger.info("Training GNN model")
        batch_size = self.gnn_params["batch_size"]
        epochs = self.gnn_params["epochs"]
        patience = self.gnn_params["patience"]
        lr = self.gnn_params["learning_rate"][0]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = self.build_gnn_model(num_node_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        with mlflow.start_run(run_name="GNN") as run:
            mlflow.log_params({
                "hidden_channels": self.gnn_params["hidden_channels"][0],
                "num_layers": self.gnn_params["num_layers"][0],
                "dropout": self.gnn_params["dropout"][0],
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
            })

            for epoch in range(1, epochs + 1):
                # Training
                model.train()
                train_loss = 0
                for batch in train_loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    pred = model(batch)
                    loss = criterion(pred, batch.y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * batch.num_graphs

                train_loss /= len(train_dataset)

                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        pred = model(batch)
                        loss = criterion(pred, batch.y)
                        val_loss += loss.item() * batch.num_graphs
                val_loss /= len(val_dataset)

                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
                )

                if epoch % 20 == 0 or epoch == 1:
                    logger.info(
                        f"Epoch {epoch}/{epochs} — "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            mlflow.log_metric("best_val_loss", best_val_loss)

        # Restore best model
        model.load_state_dict(best_model_state)
        path = "models/gnn_best.pt"
        torch.save(model.state_dict(), path)
        logger.info(f"Saved GNN model to {path}")
        return model

    # ── Utilities ─────────────────────────────────────────────────────

    @staticmethod
    def save_model(model, path: str):
        """Save any model to disk."""
        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), path)
        else:
            joblib.dump(model, path)
        logger.info(f"Saved model to {path}")

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.components.data_loader import ChEMBLDataLoader
from src.components.feature_engineering import MolecularFeatureEngineer
from src.components.model_trainer import IC50ModelTrainer
from src.components.model_evaluator import ModelEvaluator
from src.logger import logger
from src.utils.model_utils import load_config


class TrainPipeline:
    """Orchestrate the full training pipeline: data → features → train → evaluate."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)

    def run(self):
        logger.info("=" * 60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 60)

        # 1. Load & curate data
        logger.info("Stage 1: Data curation")
        loader = ChEMBLDataLoader(self.config)
        curated_df = loader.run()

        # 2. Feature engineering
        logger.info("Stage 2: Feature engineering")
        engineer = MolecularFeatureEngineer(self.config)
        feature_df = engineer.build_feature_matrix(curated_df)
        engineer.save_features(feature_df)

        graph_dataset = engineer.build_graph_dataset(
            curated_df["canonical_smiles"].tolist(),
            curated_df["pIC50"].tolist(),
        )
        engineer.save_graph_dataset(graph_dataset)

        # 3. Train/test split
        X = feature_df.values
        y = curated_df["pIC50"].values
        random_state = self.config["model"]["random_state"]
        test_size = self.config["model"]["test_size"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        indices = np.arange(len(graph_dataset))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        train_graphs = [graph_dataset[i] for i in train_idx]
        test_graphs = [graph_dataset[i] for i in test_idx]

        # 4. Train models
        logger.info("Stage 3: Model training")
        trainer = IC50ModelTrainer(self.config)

        rf_model = trainer.train_random_forest(X_train, y_train)
        xgb_model = trainer.train_xgboost(X_train, y_train)

        num_node_features = train_graphs[0].x.shape[1]
        gnn_model = trainer.train_gnn(train_graphs, test_graphs, num_node_features)

        # 5. Evaluate & compare
        logger.info("Stage 4: Model evaluation")
        evaluator = ModelEvaluator()
        device = trainer.device

        rf_pred = rf_model.predict(X_test)
        xgb_pred = xgb_model.predict(X_test)
        gnn_pred = evaluator.predict_gnn(gnn_model, test_graphs, device)
        gnn_y_true = np.array([g.y.item() for g in test_graphs])

        results = {
            "RandomForest": {"y_true": y_test, "y_pred": rf_pred},
            "XGBoost": {"y_true": y_test, "y_pred": xgb_pred},
            "GNN": {"y_true": gnn_y_true, "y_pred": gnn_pred},
        }

        evaluator.generate_report(results)

        feature_names = list(feature_df.columns)
        evaluator.plot_feature_importance(rf_model, feature_names, "RandomForest")
        evaluator.plot_feature_importance(xgb_model, feature_names, "XGBoost")

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("=" * 60)

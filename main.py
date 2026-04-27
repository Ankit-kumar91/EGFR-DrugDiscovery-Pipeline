"""CLI entry point for the EGFR Drug Discovery Pipeline.

Usage:
    python main.py --stage data         # Data curation only
    python main.py --stage features     # Feature engineering only
    python main.py --stage train        # Full training pipeline (data + features + models)
    python main.py --stage screen       # Virtual screening
    python main.py --stage dock         # Molecular docking
    python main.py --stage md           # MD simulation
    python main.py --stage all          # Run everything
    python main.py --predict SMILES     # Predict pIC50 for a single compound
"""

import argparse
import sys

from src.logger import logger
from src.exception import PipelineException


def run_data_stage(config):
    from src.components.data_loader import ChEMBLDataLoader
    loader = ChEMBLDataLoader(config)
    return loader.run()


def run_features_stage(config):
    import pandas as pd
    from src.components.feature_engineering import MolecularFeatureEngineer

    curated_df = pd.read_csv("data/processed/egfr_curated.csv")
    engineer = MolecularFeatureEngineer(config)
    feature_df = engineer.build_feature_matrix(curated_df)
    engineer.save_features(feature_df)
    graph_dataset = engineer.build_graph_dataset(
        curated_df["canonical_smiles"].tolist(),
        curated_df["pIC50"].tolist(),
    )
    engineer.save_graph_dataset(graph_dataset)
    return feature_df


def run_train_stage(config):
    from src.pipeline.train_pipeline import TrainPipeline
    pipeline = TrainPipeline()
    pipeline.run()


def run_screen_stage(config):
    from src.pipeline.predict_pipeline import PredictPipeline
    pipeline = PredictPipeline()
    return pipeline.run_virtual_screening()


def run_dock_stage(config):
    import pandas as pd
    from src.components.docking import GlideDocking

    hits = pd.read_csv("results/virtual_screening_hits.csv")
    docker = GlideDocking(config)
    smiles_list = hits["canonical_smiles"].tolist()[:config["docking"]["top_n_for_docking"]]
    return docker.run(smiles_list)


def run_md_stage(config):
    from src.components.md_simulation import GROMACSSimulation

    md_sim = GROMACSSimulation(config)
    # Write MDP files for both compounds
    for i in [1, 2]:
        outdir = f"results/md_simulation/compound_{i}"
        md_sim.write_mdp_files(outdir)
        logger.info(f"MDP files written for compound_{i}")
    logger.info(
        "MD setup complete. Run simulations with GROMACS using the generated MDP files."
    )


def predict_single(smiles: str):
    from src.pipeline.predict_pipeline import PredictPipeline
    pipeline = PredictPipeline()
    result = pipeline.predict_single(smiles)
    for key, value in result.items():
        print(f"  {key}: {value}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="EGFR Drug Discovery Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        choices=["data", "features", "train", "screen", "dock", "md", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="Predict pIC50 for a single SMILES string",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file (default: configs/config.yaml)",
    )

    args = parser.parse_args()

    if not args.stage and not args.predict:
        parser.print_help()
        return

    from src.utils.model_utils import load_config
    config = load_config(args.config)

    try:
        if args.predict:
            predict_single(args.predict)
            return

        stage = args.stage

        if stage in ("data", "all"):
            run_data_stage(config)

        if stage in ("features", "all"):
            run_features_stage(config)

        if stage == "train" or stage == "all":
            if stage == "train":
                run_train_stage(config)
            elif stage == "all":
                run_train_stage(config)

        if stage in ("screen", "all"):
            run_screen_stage(config)

        if stage in ("dock", "all"):
            run_dock_stage(config)

        if stage in ("md", "all"):
            run_md_stage(config)

        logger.info(f"Pipeline stage '{stage}' completed successfully")

    except PipelineException as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()

import os
import sys

import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors

from src.logger import logger
from src.exception import PipelineException


class ChEMBLDataLoader:
    """Fetch, clean, and curate EGFR bioactivity data from ChEMBL."""

    def __init__(self, config: dict):
        self.target_id = config["data"]["chembl_target_id"]
        self.bioactivity_type = config["data"]["bioactivity_type"]
        self.output_dir = config["data"]["output_dir"]
        os.makedirs(os.path.join(self.output_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "processed"), exist_ok=True)

    def fetch_bioactivity_data(self, target_id: str | None = None) -> pd.DataFrame:
        """Query ChEMBL for EGFR IC50 data."""
        target_id = target_id or self.target_id
        logger.info(f"Fetching bioactivity data for target {target_id}")
        try:
            activity = new_client.activity
            results = activity.filter(
                target_chembl_id=target_id,
                standard_type=self.bioactivity_type,
                standard_units="nM",
                standard_relation="=",
            ).only(
                "molecule_chembl_id",
                "canonical_smiles",
                "standard_value",
                "standard_type",
                "standard_units",
                "standard_relation",
                "pchembl_value",
                "assay_chembl_id",
                "target_chembl_id",
            )
            df = pd.DataFrame(results)
            logger.info(f"Fetched {len(df)} bioactivity records")
            return df
        except Exception as e:
            raise PipelineException(f"Failed to fetch ChEMBL data: {e}", sys)

    def preprocess_bioactivity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data, compute pIC50, and label active/inactive."""
        logger.info("Preprocessing bioactivity data")
        initial_count = len(df)

        # Drop rows with missing SMILES or standard_value
        df = df.dropna(subset=["canonical_smiles", "standard_value"]).copy()
        logger.info(f"After dropping missing SMILES/values: {len(df)} rows (removed {initial_count - len(df)})")

        # Convert standard_value to numeric
        df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
        df = df.dropna(subset=["standard_value"])

        # Remove non-positive IC50 values
        df = df[df["standard_value"] > 0]

        # Remove duplicates: keep the median IC50 per compound
        df = (
            df.groupby("molecule_chembl_id")
            .agg(
                canonical_smiles=("canonical_smiles", "first"),
                standard_value=("standard_value", "median"),
            )
            .reset_index()
        )

        # Compute pIC50 = -log10(IC50_nM * 1e-9) = 9 - log10(IC50_nM)
        df["pIC50"] = 9 - np.log10(df["standard_value"])

        # Label active vs inactive (pIC50 >= 6 → active)
        df["activity_class"] = np.where(df["pIC50"] >= 6.0, "active", "inactive")

        # Validate SMILES
        valid_mask = df["canonical_smiles"].apply(
            lambda s: Chem.MolFromSmiles(s) is not None
        )
        removed = (~valid_mask).sum()
        df = df[valid_mask].reset_index(drop=True)
        if removed > 0:
            logger.info(f"Removed {removed} compounds with invalid SMILES")

        logger.info(
            f"Curated dataset: {len(df)} compounds | "
            f"Active: {(df['activity_class'] == 'active').sum()} | "
            f"Inactive: {(df['activity_class'] == 'inactive').sum()}"
        )
        return df

    @staticmethod
    def validate_smiles(smiles_list: list[str]) -> list[str]:
        """Return only valid SMILES strings."""
        valid = []
        for smi in smiles_list:
            if smi and Chem.MolFromSmiles(smi) is not None:
                valid.append(smi)
        logger.info(f"Validated SMILES: {len(valid)}/{len(smiles_list)} valid")
        return valid

    @staticmethod
    def compute_lipinski(smiles: str) -> dict | None:
        """Compute Lipinski Rule of 5 properties for a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        return {
            "MolWt": mw,
            "LogP": logp,
            "NumHDonors": hbd,
            "NumHAcceptors": hba,
            "Lipinski_Pass": (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10),
        }

    def save_curated_data(self, df: pd.DataFrame, filename: str = "egfr_curated.csv"):
        """Save curated dataframe to processed directory."""
        path = os.path.join(self.output_dir, "processed", filename)
        df.to_csv(path, index=False)
        logger.info(f"Saved curated data to {path}")
        return path

    def save_raw_data(self, df: pd.DataFrame, filename: str = "chembl_egfr_raw.csv"):
        """Save raw dataframe to raw directory."""
        path = os.path.join(self.output_dir, "raw", filename)
        df.to_csv(path, index=False)
        logger.info(f"Saved raw data to {path}")
        return path

    def run(self) -> pd.DataFrame:
        """Execute the full data loading pipeline."""
        logger.info("Starting data curation pipeline")
        raw_df = self.fetch_bioactivity_data()
        self.save_raw_data(raw_df)
        curated_df = self.preprocess_bioactivity(raw_df)
        self.save_curated_data(curated_df)
        logger.info("Data curation pipeline complete")
        return curated_df

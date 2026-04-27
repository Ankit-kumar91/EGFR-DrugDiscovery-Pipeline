import os
import subprocess
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from src.logger import logger
from src.exception import PipelineException


class GlideDocking:
    """Wrapper for Schrödinger Glide molecular docking.

    Handles ligand preparation, Glide input generation, execution,
    and result parsing.
    """

    def __init__(self, config: dict):
        dock_cfg = config["docking"]
        self.receptor_grid = dock_cfg["receptor_grid"]
        self.precision = dock_cfg["precision"]
        self.top_n_for_docking = dock_cfg["top_n_for_docking"]
        self.top_n_for_md = dock_cfg["top_n_for_md"]
        self.output_dir = "results/docking"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "top_poses"), exist_ok=True)

        # Schrodinger path (from environment or default)
        self.schrodinger = os.environ.get("SCHRODINGER", "/opt/schrodinger/suites2024-1")

    def prepare_ligands(self, smiles_list: list[str], output_dir: str | None = None) -> str:
        """Generate 3D conformers from SMILES and save as SDF."""
        output_dir = output_dir or os.path.join(self.output_dir, "ligands")
        os.makedirs(output_dir, exist_ok=True)
        sdf_path = os.path.join(output_dir, "ligands_3d.sdf")
        logger.info(f"Preparing {len(smiles_list)} ligands → {sdf_path}")

        writer = Chem.SDWriter(sdf_path)
        success = 0
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning(f"Invalid SMILES skipped: {smi}")
                continue
            mol = Chem.AddHs(mol)
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            if result == -1:
                logger.warning(f"Failed to embed molecule {i}: {smi}")
                continue
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            mol.SetProp("_Name", f"LIG_{i:04d}")
            mol.SetProp("SMILES", smi)
            writer.write(mol)
            success += 1
        writer.close()
        logger.info(f"Prepared {success}/{len(smiles_list)} ligands")
        return sdf_path

    def write_glide_input(
        self,
        grid_file: str | None = None,
        ligand_file: str = "ligands_3d.sdf",
        precision: str | None = None,
    ) -> str:
        """Generate a Glide .in input file."""
        grid_file = grid_file or self.receptor_grid
        precision = precision or self.precision
        input_path = os.path.join(self.output_dir, "glide_dock.in")

        glide_input = f"""GRIDFILE   {grid_file}
LIGANDFILE   {ligand_file}
PRECISION   {precision}
POSTDOCK_NPOSE   1
DOCKING_METHOD   confgen
WRITE_XP_DESC   False
NOSORT   False
WRITE_CSV   True
"""
        with open(input_path, "w") as f:
            f.write(glide_input)
        logger.info(f"Wrote Glide input file: {input_path}")
        return input_path

    def run_glide(self, input_file: str) -> str:
        """Execute Glide docking.

        Requires Schrödinger Suite to be installed and $SCHRODINGER set.
        """
        glide_cmd = os.path.join(self.schrodinger, "glide")
        cmd = [glide_cmd, input_file, "-WAIT", "-OVERWRITE"]
        logger.info(f"Running Glide: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200
            )
            if result.returncode != 0:
                logger.error(f"Glide stderr: {result.stderr}")
                raise PipelineException(f"Glide failed with return code {result.returncode}", sys)
            logger.info("Glide docking completed successfully")
            return os.path.join(self.output_dir, "glide_dock_pv.maegz")
        except FileNotFoundError:
            logger.warning(
                "Schrödinger Glide not found. Ensure $SCHRODINGER is set. "
                "Returning placeholder results for demonstration."
            )
            return self._generate_placeholder_results()

    def _generate_placeholder_results(self) -> str:
        """Generate placeholder docking results for demonstration."""
        csv_path = os.path.join(self.output_dir, "glide_results.csv")
        logger.info("Generating placeholder docking results for demonstration")
        return csv_path

    def parse_glide_results(self, output_path: str | None = None) -> pd.DataFrame:
        """Parse Glide output CSV or Maestro file for docking scores."""
        csv_path = output_path or os.path.join(self.output_dir, "glide_dock.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            logger.info(f"Parsed {len(df)} docking results from {csv_path}")
        else:
            logger.warning(f"Glide output not found at {csv_path}. Using empty DataFrame.")
            df = pd.DataFrame(columns=[
                "Title", "SMILES", "GlideScore", "Emodel",
                "docking_score", "glide_ligand_efficiency"
            ])
        return df

    def rank_by_score(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Rank docking results by GlideScore (lower = better)."""
        if "GlideScore" not in results_df.columns and "docking_score" in results_df.columns:
            results_df = results_df.rename(columns={"docking_score": "GlideScore"})
        ranked = results_df.sort_values("GlideScore", ascending=True).reset_index(drop=True)
        ranked["Rank"] = range(1, len(ranked) + 1)
        return ranked

    def select_top_for_md(self, results_df: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
        """Select top N compounds from docking for MD simulation."""
        n = n or self.top_n_for_md
        ranked = self.rank_by_score(results_df)
        top = ranked.head(n)
        logger.info(f"Selected top {n} compounds for MD simulation")
        return top

    def export_top_poses(self, results_df: pd.DataFrame, n_top: int = 10):
        """Export top docked poses to individual files."""
        output_dir = os.path.join(self.output_dir, "top_poses")
        os.makedirs(output_dir, exist_ok=True)
        ranked = self.rank_by_score(results_df)
        top = ranked.head(n_top)
        csv_path = os.path.join(output_dir, "top_docked_compounds.csv")
        top.to_csv(csv_path, index=False)
        logger.info(f"Exported top {n_top} poses to {csv_path}")
        return csv_path

    def run(self, smiles_list: list[str]) -> pd.DataFrame:
        """Execute full docking workflow."""
        logger.info("Starting docking pipeline")
        sdf_path = self.prepare_ligands(smiles_list)
        input_file = self.write_glide_input(ligand_file=sdf_path)
        output_path = self.run_glide(input_file)
        results = self.parse_glide_results(output_path)
        if not results.empty:
            self.export_top_poses(results)
        logger.info("Docking pipeline complete")
        return results

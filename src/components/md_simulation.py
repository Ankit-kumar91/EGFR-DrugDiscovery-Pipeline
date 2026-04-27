import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.logger import logger
from src.exception import PipelineException


class GROMACSSimulation:
    """Wrapper for GROMACS molecular dynamics simulations.

    Handles system setup (topology, solvation, ions), energy minimization,
    equilibration (NVT/NPT), production MD, and trajectory analysis.
    """

    def __init__(self, config: dict):
        md_cfg = config["md_simulation"]
        self.force_field = md_cfg["force_field"]
        self.water_model = md_cfg["water_model"]
        self.sim_time_ns = md_cfg["simulation_time_ns"]
        self.temperature = md_cfg["temperature_k"]
        self.base_dir = "results/md_simulation"
        os.makedirs(self.base_dir, exist_ok=True)

        # GROMACS binary (gmx or gmx_mpi)
        self.gmx = os.environ.get("GMX", "gmx")

    def _run_cmd(self, cmd: list[str], input_text: str | None = None, cwd: str | None = None):
        """Run a shell command and log output."""
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                input=input_text,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=86400,  # 24h max
            )
            if result.returncode != 0:
                logger.error(f"Command failed: {result.stderr}")
                raise PipelineException(f"GROMACS command failed: {' '.join(cmd)}", sys)
            return result.stdout
        except FileNotFoundError:
            raise PipelineException(
                f"GROMACS not found. Ensure '{self.gmx}' is in PATH.", sys
            )

    def prepare_topology(self, pdb_file: str, output_dir: str) -> dict:
        """Prepare system topology with pdb2gmx, editconf, solvate, and genion.

        Returns dict with paths to key output files.
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Preparing topology for {pdb_file}")

        files = {}

        # 1. pdb2gmx — generate topology
        gro = os.path.join(output_dir, "protein.gro")
        top = os.path.join(output_dir, "topol.top")
        self._run_cmd([
            self.gmx, "pdb2gmx",
            "-f", pdb_file,
            "-o", gro,
            "-p", top,
            "-ff", self.force_field,
            "-water", self.water_model,
            "-ignh",
        ])
        files["gro"] = gro
        files["top"] = top

        # 2. editconf — define box
        box_gro = os.path.join(output_dir, "box.gro")
        self._run_cmd([
            self.gmx, "editconf",
            "-f", gro,
            "-o", box_gro,
            "-c", "-d", "1.2", "-bt", "dodecahedron",
        ])
        files["box_gro"] = box_gro

        # 3. solvate
        solv_gro = os.path.join(output_dir, "solvated.gro")
        self._run_cmd([
            self.gmx, "solvate",
            "-cp", box_gro,
            "-cs", "spc216.gro",
            "-o", solv_gro,
            "-p", top,
        ])
        files["solv_gro"] = solv_gro

        # 4. genion — add ions to neutralize
        ions_mdp = self._write_ions_mdp(output_dir)
        tpr = os.path.join(output_dir, "ions.tpr")
        self._run_cmd([
            self.gmx, "grompp",
            "-f", ions_mdp,
            "-c", solv_gro,
            "-p", top,
            "-o", tpr,
            "-maxwarn", "1",
        ])
        ions_gro = os.path.join(output_dir, "ions.gro")
        self._run_cmd(
            [self.gmx, "genion", "-s", tpr, "-o", ions_gro, "-p", top,
             "-pname", "NA", "-nname", "CL", "-neutral"],
            input_text="SOL\n",
        )
        files["ions_gro"] = ions_gro

        logger.info("Topology preparation complete")
        return files

    def prepare_ligand_topology(self, ligand_file: str, output_dir: str) -> dict:
        """Generate ligand parameters using ACPYPE (AmberTools).

        Args:
            ligand_file: Path to ligand MOL2 or PDB file.
            output_dir: Directory for output files.
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Generating ligand topology for {ligand_file}")

        try:
            self._run_cmd([
                "acpype",
                "-i", ligand_file,
                "-b", "LIG",
                "-c", "bcc",
                "-o", "gmx",
            ], cwd=output_dir)
        except PipelineException:
            logger.warning(
                "ACPYPE not found. Install with: conda install -c conda-forge acpype. "
                "Returning empty dict."
            )
            return {}

        return {
            "itp": os.path.join(output_dir, "LIG.acpype", "LIG_GMX.itp"),
            "gro": os.path.join(output_dir, "LIG.acpype", "LIG_GMX.gro"),
        }

    def write_mdp_files(self, output_dir: str) -> dict:
        """Write MDP parameter files for EM, NVT, NPT, and production."""
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        # Energy minimization
        paths["em"] = self._write_em_mdp(output_dir)
        # NVT equilibration
        paths["nvt"] = self._write_nvt_mdp(output_dir)
        # NPT equilibration
        paths["npt"] = self._write_npt_mdp(output_dir)
        # Production MD
        paths["md"] = self._write_production_mdp(output_dir)

        logger.info(f"Wrote MDP files to {output_dir}")
        return paths

    def _write_ions_mdp(self, output_dir: str) -> str:
        path = os.path.join(output_dir, "ions.mdp")
        with open(path, "w") as f:
            f.write("; Ions MDP\nintegrator = steep\nnsteps = 50000\nemtol = 1000.0\nemstep = 0.01\n")
        return path

    def _write_em_mdp(self, output_dir: str) -> str:
        path = os.path.join(output_dir, "em.mdp")
        content = """; Energy minimization
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000

nstlist     = 1
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
"""
        with open(path, "w") as f:
            f.write(content)
        return path

    def _write_nvt_mdp(self, output_dir: str) -> str:
        path = os.path.join(output_dir, "nvt.mdp")
        content = f"""; NVT equilibration
integrator  = md
nsteps      = 50000    ; 100 ps
dt          = 0.002

nstxout-compressed = 5000
nstenergy   = 5000
nstlog      = 5000

continuation = no
constraint_algorithm = lincs
constraints = h-bonds
lincs_iter  = 1
lincs_order = 4

cutoff-scheme = Verlet
ns_type     = grid
nstlist     = 10
rcoulomb    = 1.0
rvdw        = 1.0
coulombtype = PME

tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau_t       = 0.1     0.1
ref_t       = {self.temperature}   {self.temperature}

pcoupl      = no
pbc         = xyz

gen_vel     = yes
gen_temp    = {self.temperature}
gen_seed    = -1
"""
        with open(path, "w") as f:
            f.write(content)
        return path

    def _write_npt_mdp(self, output_dir: str) -> str:
        path = os.path.join(output_dir, "npt.mdp")
        content = f"""; NPT equilibration
integrator  = md
nsteps      = 50000    ; 100 ps
dt          = 0.002

nstxout-compressed = 5000
nstenergy   = 5000
nstlog      = 5000

continuation = yes
constraint_algorithm = lincs
constraints = h-bonds
lincs_iter  = 1
lincs_order = 4

cutoff-scheme = Verlet
ns_type     = grid
nstlist     = 10
rcoulomb    = 1.0
rvdw        = 1.0
coulombtype = PME

tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau_t       = 0.1     0.1
ref_t       = {self.temperature}   {self.temperature}

pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5

pbc         = xyz

gen_vel     = no
"""
        with open(path, "w") as f:
            f.write(content)
        return path

    def _write_production_mdp(self, output_dir: str) -> str:
        path = os.path.join(output_dir, "md.mdp")
        nsteps = int(self.sim_time_ns * 1e6 / 2)  # dt=0.002 ps
        content = f"""; Production MD
integrator  = md
nsteps      = {nsteps}    ; {self.sim_time_ns} ns
dt          = 0.002

nstxout-compressed = 5000
nstenergy   = 5000
nstlog      = 5000

continuation = yes
constraint_algorithm = lincs
constraints = h-bonds
lincs_iter  = 1
lincs_order = 4

cutoff-scheme = Verlet
ns_type     = grid
nstlist     = 10
rcoulomb    = 1.0
rvdw        = 1.0
coulombtype = PME

tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau_t       = 0.1     0.1
ref_t       = {self.temperature}   {self.temperature}

pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5

pbc         = xyz

gen_vel     = no
"""
        with open(path, "w") as f:
            f.write(content)
        return path

    def run_energy_minimization(self, topology_dir: str, files: dict) -> str:
        """Run energy minimization step."""
        logger.info("Running energy minimization")
        em_mdp = os.path.join(topology_dir, "em.mdp")
        tpr = os.path.join(topology_dir, "em.tpr")
        self._run_cmd([
            self.gmx, "grompp",
            "-f", em_mdp,
            "-c", files["ions_gro"],
            "-p", files["top"],
            "-o", tpr,
            "-maxwarn", "1",
        ])
        self._run_cmd([self.gmx, "mdrun", "-deffnm", os.path.join(topology_dir, "em")])
        logger.info("Energy minimization complete")
        return os.path.join(topology_dir, "em.gro")

    def run_equilibration(self, topology_dir: str, files: dict, em_gro: str) -> str:
        """Run NVT and NPT equilibration."""
        # NVT
        logger.info("Running NVT equilibration")
        nvt_tpr = os.path.join(topology_dir, "nvt.tpr")
        self._run_cmd([
            self.gmx, "grompp",
            "-f", os.path.join(topology_dir, "nvt.mdp"),
            "-c", em_gro,
            "-r", em_gro,
            "-p", files["top"],
            "-o", nvt_tpr,
            "-maxwarn", "1",
        ])
        self._run_cmd([self.gmx, "mdrun", "-deffnm", os.path.join(topology_dir, "nvt")])

        # NPT
        logger.info("Running NPT equilibration")
        npt_tpr = os.path.join(topology_dir, "npt.tpr")
        nvt_gro = os.path.join(topology_dir, "nvt.gro")
        self._run_cmd([
            self.gmx, "grompp",
            "-f", os.path.join(topology_dir, "npt.mdp"),
            "-c", nvt_gro,
            "-r", nvt_gro,
            "-p", files["top"],
            "-t", os.path.join(topology_dir, "nvt.cpt"),
            "-o", npt_tpr,
            "-maxwarn", "1",
        ])
        self._run_cmd([self.gmx, "mdrun", "-deffnm", os.path.join(topology_dir, "npt")])

        logger.info("Equilibration complete")
        return os.path.join(topology_dir, "npt.gro")

    def run_production(self, topology_dir: str, files: dict, npt_gro: str) -> str:
        """Run production MD simulation."""
        logger.info(f"Running production MD ({self.sim_time_ns} ns)")
        md_tpr = os.path.join(topology_dir, "md.tpr")
        self._run_cmd([
            self.gmx, "grompp",
            "-f", os.path.join(topology_dir, "md.mdp"),
            "-c", npt_gro,
            "-p", files["top"],
            "-t", os.path.join(topology_dir, "npt.cpt"),
            "-o", md_tpr,
            "-maxwarn", "1",
        ])
        self._run_cmd([self.gmx, "mdrun", "-deffnm", os.path.join(topology_dir, "md")])
        logger.info("Production MD complete")
        return os.path.join(topology_dir, "md.xtc")

    # ── Analysis ──────────────────────────────────────────────────────

    def analyze_rmsd(self, trajectory: str, topology: str, output_dir: str) -> pd.DataFrame:
        """Compute RMSD over the trajectory."""
        xvg_path = os.path.join(output_dir, "rmsd.xvg")
        self._run_cmd(
            [self.gmx, "rms", "-s", topology, "-f", trajectory, "-o", xvg_path, "-tu", "ns"],
            input_text="Backbone\nBackbone\n",
        )
        return self._parse_xvg(xvg_path, columns=["Time_ns", "RMSD_nm"])

    def analyze_rmsf(self, trajectory: str, topology: str, output_dir: str) -> pd.DataFrame:
        """Compute RMSF per residue."""
        xvg_path = os.path.join(output_dir, "rmsf.xvg")
        self._run_cmd(
            [self.gmx, "rmsf", "-s", topology, "-f", trajectory, "-o", xvg_path, "-res"],
            input_text="Backbone\n",
        )
        return self._parse_xvg(xvg_path, columns=["Residue", "RMSF_nm"])

    def analyze_rg(self, trajectory: str, topology: str, output_dir: str) -> pd.DataFrame:
        """Compute radius of gyration."""
        xvg_path = os.path.join(output_dir, "gyrate.xvg")
        self._run_cmd(
            [self.gmx, "gyrate", "-s", topology, "-f", trajectory, "-o", xvg_path],
            input_text="Protein\n",
        )
        return self._parse_xvg(xvg_path, columns=["Time_ps", "Rg_nm", "RgX", "RgY", "RgZ"])

    def analyze_hbonds(self, trajectory: str, topology: str, output_dir: str) -> pd.DataFrame:
        """Analyze hydrogen bonds between protein and ligand."""
        xvg_path = os.path.join(output_dir, "hbonds.xvg")
        self._run_cmd(
            [self.gmx, "hbond", "-s", topology, "-f", trajectory, "-num", xvg_path],
            input_text="Protein\nLIG\n",
        )
        return self._parse_xvg(xvg_path, columns=["Time_ps", "Num_HBonds"])

    @staticmethod
    def _parse_xvg(path: str, columns: list[str]) -> pd.DataFrame:
        """Parse a GROMACS .xvg file into a DataFrame."""
        data = []
        if not os.path.exists(path):
            logger.warning(f"XVG file not found: {path}")
            return pd.DataFrame(columns=columns)
        with open(path) as f:
            for line in f:
                if line.startswith(("#", "@")):
                    continue
                values = line.strip().split()
                if values:
                    data.append([float(v) for v in values])
        df = pd.DataFrame(data)
        if len(df.columns) >= len(columns):
            df = df.iloc[:, : len(columns)]
            df.columns = columns
        return df

    # ── Plotting ──────────────────────────────────────────────────────

    @staticmethod
    def plot_rmsd(rmsd_df: pd.DataFrame, title: str = "RMSD", output_path: str | None = None):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rmsd_df["Time_ns"], rmsd_df["RMSD_nm"], color="steelblue", linewidth=0.8)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("RMSD (nm)")
        ax.set_title(title)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_rmsf(rmsf_df: pd.DataFrame, title: str = "RMSF", output_path: str | None = None):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(rmsf_df["Residue"], rmsf_df["RMSF_nm"], color="steelblue", linewidth=0.8)
        ax.set_xlabel("Residue Number")
        ax.set_ylabel("RMSF (nm)")
        ax.set_title(title)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_rg(rg_df: pd.DataFrame, title: str = "Radius of Gyration", output_path: str | None = None):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rg_df["Time_ps"] / 1000, rg_df["Rg_nm"], color="steelblue", linewidth=0.8)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Rg (nm)")
        ax.set_title(title)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_energy(edr_file: str, output_path: str | None = None):
        """Plot energy from .edr file (requires gmx energy)."""
        logger.info("Energy plotting requires gmx energy — skipping if GROMACS not available")

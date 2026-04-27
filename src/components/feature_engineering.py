import os
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors
from torch_geometric.data import Data

from src.logger import logger
from src.exception import PipelineException

# Atom feature dimensions
ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-2, -1, 0, 1, 2],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    "is_aromatic": [False, True],
}

BOND_TYPES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


def one_hot(value, choices: list) -> list[int]:
    """One-hot encode a value from a list of choices."""
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    return encoding


class MolecularFeatureEngineer:
    """Generate molecular fingerprints, descriptors, and graph representations."""

    def __init__(self, config: dict):
        feat_cfg = config["features"]
        self.fp_type = feat_cfg["fingerprint_type"]
        self.fp_radius = feat_cfg["fp_radius"]
        self.fp_nbits = feat_cfg["fp_nbits"]
        self.descriptor_names = feat_cfg["descriptors"]
        self.output_dir = config["data"]["output_dir"]

    def compute_fingerprints(
        self,
        smiles_list: list[str],
        fp_type: str | None = None,
        radius: int | None = None,
        nbits: int | None = None,
    ) -> np.ndarray:
        """Compute molecular fingerprints for a list of SMILES."""
        fp_type = fp_type or self.fp_type
        radius = radius or self.fp_radius
        nbits = nbits or self.fp_nbits
        logger.info(f"Computing {fp_type} fingerprints (radius={radius}, nbits={nbits})")

        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fps.append(np.zeros(nbits))
                continue
            if fp_type == "morgan":
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            elif fp_type == "maccs":
                fp = MACCSkeys.GenMACCSKeys(mol)
            elif fp_type == "rdkit":
                fp = Chem.RDKFingerprint(mol, fpSize=nbits)
            else:
                raise PipelineException(f"Unknown fingerprint type: {fp_type}", sys)
            arr = np.zeros(len(fp))
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)

        return np.array(fps)

    def compute_descriptors(self, smiles_list: list[str]) -> pd.DataFrame:
        """Compute RDKit molecular descriptors."""
        logger.info(f"Computing descriptors: {self.descriptor_names}")
        descriptor_map = {
            "MolWt": Descriptors.ExactMolWt,
            "LogP": Descriptors.MolLogP,
            "NumHDonors": Descriptors.NumHDonors,
            "NumHAcceptors": Descriptors.NumHAcceptors,
            "TPSA": Descriptors.TPSA,
            "NumRotatableBonds": Descriptors.NumRotatableBonds,
        }
        records = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                records.append({name: np.nan for name in self.descriptor_names})
                continue
            row = {}
            for name in self.descriptor_names:
                func = descriptor_map.get(name)
                if func is None:
                    raise PipelineException(f"Unknown descriptor: {name}", sys)
                row[name] = func(mol)
            records.append(row)
        return pd.DataFrame(records)

    def compute_lipinski(self, smiles_list: list[str]) -> pd.DataFrame:
        """Compute Lipinski Rule of 5 compliance."""
        records = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                records.append({"Lipinski_Pass": False})
                continue
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
            records.append({
                "MolWt": mw,
                "LogP": logp,
                "NumHDonors": hbd,
                "NumHAcceptors": hba,
                "Lipinski_Violations": violations,
                "Lipinski_Pass": violations == 0,
            })
        return pd.DataFrame(records)

    @staticmethod
    def smiles_to_graph(smiles: str, y: float | None = None) -> Data | None:
        """Convert a SMILES string to a PyTorch Geometric Data object."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Node (atom) features
        node_features = []
        for atom in mol.GetAtoms():
            features = (
                one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
                + one_hot(atom.GetDegree(), ATOM_FEATURES["degree"])
                + one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
                + one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
                + one_hot(atom.GetIsAromatic(), ATOM_FEATURES["is_aromatic"])
                + [atom.GetMass() / 100.0]  # normalized mass
            )
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)

        # Edge index and edge features
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_features = [
                BOND_TYPES.get(bond.GetBondType(), 0),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
            ]
            # Undirected: add both directions
            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([bond_features, bond_features])

        if len(edge_index) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if y is not None:
            data.y = torch.tensor([y], dtype=torch.float)
        return data

    def build_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine fingerprints and descriptors into a single feature matrix."""
        logger.info("Building combined feature matrix (fingerprints + descriptors)")
        smiles_list = df["canonical_smiles"].tolist()

        # Fingerprints
        fp_array = self.compute_fingerprints(smiles_list)
        fp_cols = [f"fp_{i}" for i in range(fp_array.shape[1])]
        fp_df = pd.DataFrame(fp_array, columns=fp_cols)

        # Descriptors
        desc_df = self.compute_descriptors(smiles_list)

        feature_df = pd.concat([fp_df, desc_df], axis=1)
        logger.info(f"Feature matrix shape: {feature_df.shape}")
        return feature_df

    def build_graph_dataset(
        self, smiles_list: list[str], labels: list[float] | None = None
    ) -> list[Data]:
        """Build a list of PyG Data objects from SMILES and optional labels."""
        logger.info(f"Building graph dataset for {len(smiles_list)} molecules")
        dataset = []
        skipped = 0
        for i, smi in enumerate(smiles_list):
            y = labels[i] if labels is not None else None
            data = self.smiles_to_graph(smi, y)
            if data is not None:
                dataset.append(data)
            else:
                skipped += 1
        if skipped > 0:
            logger.warning(f"Skipped {skipped} molecules with invalid SMILES")
        logger.info(f"Graph dataset size: {len(dataset)}")
        return dataset

    def save_features(self, feature_df: pd.DataFrame, filename: str = "egfr_features.csv"):
        """Save feature matrix to CSV."""
        path = os.path.join(self.output_dir, "processed", filename)
        feature_df.to_csv(path, index=False)
        logger.info(f"Saved feature matrix to {path}")
        return path

    def save_graph_dataset(self, dataset: list[Data], filename: str = "egfr_graphs.pt"):
        """Save graph dataset to disk."""
        path = os.path.join(self.output_dir, "processed", filename)
        torch.save(dataset, path)
        logger.info(f"Saved graph dataset ({len(dataset)} graphs) to {path}")
        return path

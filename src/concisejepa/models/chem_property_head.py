import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False

# Each entry: (property_name, bin_edges)
# bin_edges define N+1 boundaries → N bins; interior boundaries used for digitize.
# Chosen to reflect drug-likeness ranges and provide reasonably balanced bins over BindingDB.
PROPERTY_SPECS: list[tuple[str, list[float]]] = [
    ("logP",          [-10.0, -2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]),  # 8 bins
    ("MW",            [0.0, 200.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1200.0]),  # 7 bins
    ("HBD",           [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, 15.0]),  # 6 bins  (0, 1, 2, 3, 4, 5+)
    ("HBA",           [-0.5, 1.5, 3.5, 5.5, 7.5, 10.5, 20.0]),  # 6 bins  (0-1, 2-3, 4-5, 6-7, 8-10, 11+)
    ("TPSA",          [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 130.0, 300.0]),  # 7 bins
    ("RotBonds",      [-0.5, 0.5, 2.5, 4.5, 6.5, 9.5, 20.0]),  # 6 bins  (0, 1-2, 3-4, 5-6, 7-9, 10+)
    ("AromaticRings", [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, 10.0]),  # 6 bins  (0, 1, 2, 3, 4, 5+)
]

_PROP_NAMES: list[str] = [name for name, _ in PROPERTY_SPECS]
_N_BINS: list[int] = [len(edges) - 1 for _, edges in PROPERTY_SPECS]
_BIN_EDGES: list[np.ndarray] = [np.array(edges[1:-1], dtype=np.float64) for _, edges in PROPERTY_SPECS]


def compute_property_bins(smiles: str) -> torch.Tensor:
    """
    Compute binned physicochemical property indices for one SMILES string.

    Returns a [num_properties] int64 tensor of bin indices.
    A value of -1 means the property could not be computed (invalid mol or RDKit error).
    """
    if not _RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required for chemical property supervision")

    result = torch.full((len(PROPERTY_SPECS),), -1, dtype=torch.long)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return result

    try:
        raw_values = [
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
        ]
    except Exception:
        return result

    for i, (val, edges) in enumerate(zip(raw_values, _BIN_EDGES)):
        result[i] = int(np.digitize(val, bins=edges))

    return result


class ChemPropertyHead(nn.Module):
    """
    Auxiliary classification heads predicting binned chemical properties
    from the drug pre-quantized representation [B, latent_dim].

    One linear head per property; all share the same input.
    Used as a training-time supervision signal to push the FSQ pre-quantized
    space (and through it, the code assignments) to align with chemical semantics.
    Not needed at inference time.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.property_names = _PROP_NAMES
        self.heads = nn.ModuleList([
            nn.Linear(in_dim, n_bins) for n_bins in _N_BINS
        ])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: [B, in_dim]  →  {prop_name: logits [B, n_bins]}"""
        return {name: head(x) for name, head in zip(self.property_names, self.heads)}

    def loss(
        self,
        x: torch.Tensor,
        smiles_list: list[str],
        property_cache: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute mean cross-entropy loss over all properties and valid molecules.

        Args:
            x: [B, in_dim] pre-quantized drug representations
            smiles_list: SMILES strings for the batch (length B)
            property_cache: dict mapping SMILES → [num_properties] int64 bin-index tensor

        Returns:
            total_loss: mean over valid properties
            per_prop_losses: {prop_name: scalar loss} for logging
        """
        missing = torch.full((len(self.property_names),), -1, dtype=torch.long)
        labels = torch.stack(
            [property_cache.get(smi, missing) for smi in smiles_list]
        ).to(x.device)  # [B, num_props]

        logits_dict = self(x)
        per_prop: dict[str, torch.Tensor] = {}
        total = x.new_zeros(())
        n_valid = 0

        for prop_idx, name in enumerate(self.property_names):
            prop_labels = labels[:, prop_idx]  # [B]
            valid = prop_labels >= 0
            if not valid.any():
                continue
            loss_i = F.cross_entropy(logits_dict[name][valid], prop_labels[valid])
            per_prop[name] = loss_i
            total = total + loss_i
            n_valid += 1

        return (total / max(n_valid, 1)), per_prop

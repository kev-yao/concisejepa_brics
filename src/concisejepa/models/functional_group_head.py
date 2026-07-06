"""
Functional group multi-label supervision head (Option A).

Forces the pre-quantized drug representation to encode the presence/absence
of ~40 drug-relevant functional groups, making FSQ code assignments
chemically interpretable.

Groups are detected via RDKit SMARTS. Any pattern that fails to compile at
import time is silently excluded, so the set is robust to RDKit version
differences.

Loss: binary cross-entropy (each group is an independent binary label).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from rdkit import Chem
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


# (display_name, SMARTS)
_FUNCTIONAL_GROUP_SPECS: list[tuple[str, str]] = [
    # Oxygen-containing
    ("alcohol",         "[OX2H1][CX4]"),
    ("phenol",          "[OX2H1]c"),
    ("carboxylic_acid", "[CX3](=O)[OX2H1]"),
    ("ester",           "[#6][CX3](=O)[OX2H0][#6]"),
    ("ketone",          "[#6][CX3](=O)[#6]"),
    ("aldehyde",        "[CX3H1](=O)"),
    ("ether",           "[OD2]([#6])[#6]"),
    # Nitrogen-containing
    ("primary_amine",   "[NX3;H2;!$(NC=O)]"),
    ("secondary_amine", "[NX3;H1;!$(NC=O)]"),
    ("tertiary_amine",  "[NX3;H0;!$(NC=O);!$(N~[!#6])]"),
    ("amide",           "[NX3][CX3](=[OX1])"),
    ("sulfonamide",     "[SX4](=[OX1])(=[OX1])[NX3]"),
    ("urea",            "[NX3][CX3](=[OX1])[NX3]"),
    ("guanidine",       "[NX3][CX3](=[NX2])[NX3]"),
    ("nitrile",         "[CX2]#N"),
    ("nitro",           "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("imine",           "[CX3]=[NX2]"),
    ("carbamate",       "[NX3][CX3](=[OX1])[OX2H0]"),
    # Sulfur-containing
    ("thiol",           "[SX2H]"),
    ("thioether",       "[SX2]([#6])[#6]"),
    ("sulfone",         "[SX4](=[OX1])(=[OX1])([#6])[#6]"),
    # Halogens
    ("fluorine",        "[F][#6]"),
    ("chlorine",        "[Cl][#6]"),
    ("bromine",         "[Br][#6]"),
    ("iodine",          "[I][#6]"),
    ("trifluoromethyl", "[CX4](F)(F)F"),
    # Aromatic rings
    ("benzene",         "c1ccccc1"),
    ("pyridine",        "n1ccccc1"),
    ("imidazole",       "c1c[nH]cn1"),
    ("pyrazole",        "c1cc[nH]n1"),
    ("thiophene",       "c1ccsc1"),
    ("furan",           "c1ccoc1"),
    ("indole",          "c1ccc2[nH]ccc2c1"),
    # Saturated N-rings (common drug scaffolds)
    ("piperidine",      "N1CCCCC1"),
    ("piperazine",      "N1CCNCC1"),
    ("morpholine",      "N1CCOCC1"),
    ("pyrrolidine",     "N1CCCC1"),
    # Other drug-like
    ("lactam",          "[NX3;r][CX3;r](=[OX1])"),
    ("lactone",         "[OX2;r][CX3;r](=[OX1])"),
    ("phosphate",       "[PX4](=[OX1])[OX2]"),
    ("hydroxamic_acid", "[CX3](=[OX1])[NX3][OX2H]"),
]

# Pre-compile SMARTS at module load; skip any that fail
_COMPILED_GROUPS: list[tuple[str, object]] = []
if _RDKIT_AVAILABLE:
    for _name, _smarts in _FUNCTIONAL_GROUP_SPECS:
        try:
            _pat = Chem.MolFromSmarts(_smarts)
            if _pat is not None:
                _COMPILED_GROUPS.append((_name, _pat))
        except Exception:
            pass

GROUP_NAMES: list[str] = [name for name, _ in _COMPILED_GROUPS]
N_GROUPS: int = len(_COMPILED_GROUPS)


def compute_functional_group_bits(smiles: str) -> torch.Tensor:
    """
    Returns a [N_GROUPS] float32 tensor (0.0 / 1.0) indicating which
    functional groups are present in the molecule.
    Returns all zeros for invalid SMILES or when RDKit is unavailable.
    """
    result = torch.zeros(N_GROUPS, dtype=torch.float32)
    if not _RDKIT_AVAILABLE or not _COMPILED_GROUPS:
        return result
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return result
    for i, (_, pattern) in enumerate(_COMPILED_GROUPS):
        try:
            if mol.HasSubstructMatch(pattern):
                result[i] = 1.0
        except Exception:
            pass
    return result


class FunctionalGroupHead(nn.Module):
    """
    Multi-label head: predicts presence/absence of each functional group
    from the pre-quantized drug representation.

    Input:  x  [B, in_dim]
    Output: logits  [B, N_GROUPS]
    Loss:   binary cross-entropy averaged over groups and batch.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.n_groups = N_GROUPS
        self.group_names = GROUP_NAMES
        self.head = nn.Linear(in_dim, N_GROUPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # [B, N_GROUPS]

    def loss(
        self,
        x: torch.Tensor,
        smiles_list: list[str],
        group_cache: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            x:           [B, in_dim] pre-quantized drug representations
            smiles_list: SMILES strings for the batch
            group_cache: smiles → [N_GROUPS] float32 tensor (0/1)

        Returns:
            loss:         scalar BCE loss
            metrics:      {"accuracy": batch-level binary accuracy scalar}
        """
        _zero = torch.zeros(self.n_groups, dtype=torch.float32)
        targets = torch.stack(
            [group_cache.get(smi, _zero) for smi in smiles_list]
        ).to(x.device)  # [B, N_GROUPS]

        logits = self(x)  # [B, N_GROUPS]
        loss = F.binary_cross_entropy_with_logits(logits, targets)

        with torch.no_grad():
            accuracy = ((logits > 0).float() == targets).float().mean()

        return loss, {"accuracy": accuracy}

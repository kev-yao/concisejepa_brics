"""
qa/test_fragment_pooling.py — Shape and gradient tests for all 5 pooling strategies.
Run on login node (CPU-only):
    cd ~/projects/concisejepa
    python -m pytest qa/test_fragment_pooling.py -v
"""

import sys
from pathlib import Path

try:
    import pytest
    _PYTEST = True
except ImportError:
    # Stub pytest decorators so the module can be imported without pytest installed.
    import types
    _PYTEST = False
    pytest = types.ModuleType("pytest")
    class _Mark:
        def parametrize(self, *a, **kw): return lambda f: f
        def __call__(self, *a, **kw): return lambda f: f
        def __getattr__(self, name): return self
    pytest.mark = _Mark()
    pytest.fixture = lambda *a, **kw: (lambda f: f)
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from spikes.phase1.fragment_pooling import (
    MeanPool, MaxPool, WeightedSumPool, CrossAttentionPool, LatentQueryPool,
    build_pooling,
)
from spikes.phase1.fragment_encoder import ConciseFragment, ConciseJEPAFragment

B, F, D = 2, 5, 128          # batch=2, max_frags=5, latent_dim=128
PROJ_DIM = 256
RESIDUE_DIM = 1280
LIGAND_DIM = 64              # small for fast test


def make_inputs(requires_grad=True):
    frag_embs = torch.randn(B, F, D, requires_grad=requires_grad)
    mask = torch.ones(B, F, dtype=torch.bool)
    mask[0, 3:] = False   # molecule 0 has 3 valid fragments
    mask[1, 4:] = False   # molecule 1 has 4 valid fragments
    protein = torch.randn(B, 50, RESIDUE_DIM)
    return frag_embs, mask, protein


# ---------------------------------------------------------------------------
# Pooling module shape tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pooler_name", ["mean", "max", "weighted_sum", "latent_query"])
def test_pooler_output_shape(pooler_name):
    pooler = build_pooling(pooler_name, dim=D)
    frag_embs, mask, _ = make_inputs(requires_grad=False)
    out = pooler(frag_embs, mask)
    assert out.shape == (B, D), f"{pooler_name}: expected ({B},{D}), got {out.shape}"


def test_cross_attention_pool_output_shape():
    pooler = CrossAttentionPool(dim=D, residue_dim=RESIDUE_DIM, nheads=4)
    frag_embs, mask, protein = make_inputs(requires_grad=False)
    out = pooler(frag_embs, mask, context=protein)
    assert out.shape == (B, D), f"cross_attention: expected ({B},{D}), got {out.shape}"


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pooler_name", ["mean", "max", "weighted_sum", "latent_query"])
def test_pooler_gradient_flow(pooler_name):
    pooler = build_pooling(pooler_name, dim=D)
    frag_embs, mask, _ = make_inputs(requires_grad=True)
    out = pooler(frag_embs, mask)
    out.sum().backward()
    assert frag_embs.grad is not None, f"{pooler_name}: gradient did not flow to inputs"
    assert not torch.isnan(frag_embs.grad).any(), f"{pooler_name}: NaN in gradients"


def test_cross_attention_pool_gradient_flow():
    pooler = CrossAttentionPool(dim=D, residue_dim=RESIDUE_DIM, nheads=4)
    frag_embs, mask, protein = make_inputs(requires_grad=True)
    protein = protein.requires_grad_(True)
    out = pooler(frag_embs, mask, context=protein)
    out.sum().backward()
    assert frag_embs.grad is not None
    assert protein.grad is not None


# ---------------------------------------------------------------------------
# Masking correctness: padding fragments should not affect output
# ---------------------------------------------------------------------------

def test_mean_pool_masking():
    pooler = MeanPool()
    frag_embs = torch.randn(1, 4, D)
    mask = torch.tensor([[True, True, False, False]])  # only 2 valid

    out_2 = pooler(frag_embs, mask)

    # Overwrite padding with garbage; output should be identical
    frag_embs_mod = frag_embs.clone()
    frag_embs_mod[0, 2:] = 999.0
    out_2_mod = pooler(frag_embs_mod, mask)

    assert torch.allclose(out_2, out_2_mod), "MeanPool: padding affected output"


def test_weighted_sum_pool_masking():
    pooler = WeightedSumPool(dim=D)
    frag_embs = torch.randn(1, 4, D)
    mask = torch.tensor([[True, True, False, False]])

    with torch.no_grad():
        out_orig = pooler(frag_embs, mask)
        frag_embs_mod = frag_embs.clone()
        frag_embs_mod[0, 2:] = 999.0
        out_mod = pooler(frag_embs_mod, mask)

    assert torch.allclose(out_orig, out_mod, atol=1e-5), "WeightedSumPool: padding affected output"


# ---------------------------------------------------------------------------
# ConciseFragment forward shape test (no DrugEncoder weights needed — uses tiny model)
# ---------------------------------------------------------------------------

def test_concise_fragment_forward_shape():
    """ConciseFragment forward pass: check output shapes."""
    tiny_layers = [[4, 4, 4]]  # small FSQ for fast test
    model = ConciseFragment(
        drug_layers=tiny_layers,
        pooling="mean",
        ligand_dim=LIGAND_DIM,
        residue_dim=RESIDUE_DIM,
        drug_dim=D,
        proj_dim=PROJ_DIM,
        nheads=8,  # must divide proj_dim
        activation="tanh",
        pairwise_attention_chunk_size=10,
        use_pairwise_attention_checkpoint=False,
    )
    model.eval()
    frag_fps = torch.randn(B, F, LIGAND_DIM)
    mask = torch.ones(B, F, dtype=torch.bool)
    mask[0, 3:] = False
    r_emb = torch.randn(B, 50, RESIDUE_DIM)

    with torch.no_grad():
        out = model(frag_fps, mask, r_emb)

    assert out["binding"].shape == (B,)
    assert out["d_emb"].shape == (B, PROJ_DIM)
    assert out["r_emb"].shape == (B, PROJ_DIM)
    assert out["pairwise_binding"].shape == (B, B)


@pytest.mark.parametrize("pooling", ["mean", "max", "weighted_sum", "cross_attention", "latent_query"])
def test_concise_fragment_all_poolings(pooling):
    tiny_layers = [[4, 4, 4]]
    model = ConciseFragment(
        drug_layers=tiny_layers,
        pooling=pooling,
        ligand_dim=LIGAND_DIM,
        residue_dim=RESIDUE_DIM,
        drug_dim=D,
        proj_dim=PROJ_DIM,
        nheads=8,
        activation="tanh",
        pairwise_attention_chunk_size=10,
        use_pairwise_attention_checkpoint=False,
    )
    model.eval()
    frag_fps = torch.randn(B, F, LIGAND_DIM)
    mask = torch.ones(B, F, dtype=torch.bool)
    r_emb = torch.randn(B, 50, RESIDUE_DIM)

    with torch.no_grad():
        out = model(frag_fps, mask, r_emb)

    assert out["binding"].shape == (B,), f"{pooling}: binding shape wrong"
    assert not torch.isnan(out["binding"]).any(), f"{pooling}: NaN in binding"


if __name__ == "__main__":
    # Quick smoke test without pytest
    test_concise_fragment_all_poolings("mean")
    test_concise_fragment_all_poolings("max")
    test_concise_fragment_all_poolings("weighted_sum")
    test_concise_fragment_all_poolings("cross_attention")
    test_concise_fragment_all_poolings("latent_query")
    print("All smoke tests passed.")

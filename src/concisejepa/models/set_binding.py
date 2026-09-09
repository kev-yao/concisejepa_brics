"""Live FSQ/F2R features and a jointly trained primary Set Transformer readout."""

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from .fragment import ConciseFragment
from .secondary_binding import ConciseJEPASecondaryBinding
from .set_binding_head import SetTransformerBindingHead


class ConciseFragmentSetReadout(ConciseFragment):
    """Replace only the primary scorer, retaining pair-conditioned F2R and FSQ.

    Whole inputs still use ``final`` and never enter this head or fragment JEPA.
    Shared upstream parameters remain trainable in both branches. The head acts
    on every candidate pair, not just the diagonal of a legacy score matrix.
    """

    def __init__(self, *args, set_head=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.d_encoder.quantizer_type != "fsq":
            raise ValueError("End-to-end Set readout requires discrete fsq")
        self.set_head = SetTransformerBindingHead(
            feature_dim=self.r_project.out_features, **dict(set_head or {})
        )

    def _score_fragment_aligned_embeddings(self, d_pair, r_pair):
        # Same four roles as the frozen-feature experiment, computed live.
        # FSQ's integer code IDs are metadata; its STE embedding stays in graph.
        drug = F.normalize(d_pair.flatten(1), dim=-1)
        protein = F.normalize(r_pair, dim=-1)
        tokens = torch.stack((drug, protein, drug * protein, (drug - protein).abs()), dim=1)
        return self.set_head(tokens)["binding"]


class ConciseJEPASetBinding(ConciseJEPASecondaryBinding):
    """Secondary-binding model with live primary Set scores and fragment JEPA."""

    def __init__(self, concise_fragment, **kwargs):
        if not isinstance(concise_fragment, ConciseFragmentSetReadout):
            raise ValueError("ConciseJEPASetBinding requires ConciseFragmentSetReadout")
        super().__init__(concise_fragment, **kwargs)

    def initialize_backbone(self, checkpoint_path):
        """Warm-start from a trusted legacy secondary-binding Lightning checkpoint.

        Only the new Set head may be absent. Unexpected/missing backbone keys or
        shape mismatches fail rather than silently leaving random parameters.
        This initializes weights, not optimizer/epoch state, and freezes nothing.
        Configuration must match the original FSQ levels and architecture: FSQ
        mode/levels are not fully described by a state dict alone.
        """
        path = Path(checkpoint_path).expanduser().resolve()
        # Lightning checkpoints may contain configuration objects. Only load
        # trusted local artifacts (torch.load with weights_only=False).
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            sha256 = digest.hexdigest()
            stream.seek(0)
            checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
        state = {key.removeprefix("model."): value
                 for key, value in checkpoint["state_dict"].items() if key.startswith("model.")}
        current = self.state_dict()
        head_keys = {key for key in current if key.startswith("concise.set_head.")}
        expected = set(current) - head_keys
        if set(state) != expected:
            raise ValueError(f"Backbone checkpoint keys differ: missing={sorted(expected - set(state))}, "
                             f"unexpected={sorted(set(state) - expected)}")
        mismatched = [key for key in expected if state[key].shape != current[key].shape]
        if mismatched:
            raise ValueError(f"Backbone checkpoint shapes differ: {sorted(mismatched)}")
        self.load_state_dict({**current, **state}, strict=True)
        return {"backbone_checkpoint": str(path), "sha256": sha256,
                "loaded_tensors": len(state), "fresh_head_tensors": len(head_keys)}


__all__ = ["ConciseFragmentSetReadout", "ConciseJEPASetBinding"]

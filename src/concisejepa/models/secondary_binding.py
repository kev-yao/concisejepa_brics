"""Corrected fragment JEPA with a separately scored whole-molecule binding view."""

import math

from .fragment import ConciseJEPAFragment


class ConciseJEPASecondaryBinding(ConciseJEPAFragment):
    """Share the encoder/scorer, never the fragment pool or JEPA input.

    Both views produce sigmoid pair scores. ``whole_binding_weight`` mixes these
    scores after scoring; it does not mix features or alter branch supervision.
    Keep the fusion weight in the resolved config when reloading a checkpoint.
    """

    def __init__(self, concise_fragment, smiles_target_dim=256, jepa_hidden_dim=512,
                 clip_logit_scale_init=14.0, whole_binding_weight=0.25):
        alpha = float(whole_binding_weight)
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("whole_binding_weight must be finite and in [0, 1]")
        if concise_fragment.d_encoder.quantizer_type not in ("fsq", "continuous_fsq"):
            raise ValueError("Secondary binding supports fsq and continuous_fsq quantizers")
        super().__init__(concise_fragment, smiles_target_dim, jepa_hidden_dim, clip_logit_scale_init)
        self.whole_binding_weight = alpha

    def forward(self, protein_embedding, frag_fps, frag_mask, whole_molecule_fingerprint=None):
        if whole_molecule_fingerprint is None:
            raise ValueError("Secondary binding requires a separate whole_molecule_fingerprint input")
        if whole_molecule_fingerprint.shape != (frag_fps.shape[0], frag_fps.shape[-1]):
            raise ValueError("whole_molecule_fingerprint must have shape [B, fingerprint_dim]")
        dti = self.concise(frag_fps, frag_mask, protein_embedding, whole_molecule_fingerprint)
        outputs = self._jepa_outputs(dti)
        alpha = self.whole_binding_weight
        for key in ("binding", "similarity_cosines", "similarity_logits"):
            outputs[f"fragment_{key}"] = outputs[key]
        outputs["whole_binding"] = dti["whole_binding"]
        outputs["whole_similarity_cosines"] = dti["whole_pairwise_binding"]
        outputs["whole_similarity_logits"] = self.logit_scale.exp() * dti["whole_pairwise_binding"]
        outputs["whole_codes"] = dti["whole_codes"]
        for key in ("binding", "similarity_cosines", "similarity_logits"):
            outputs[key] = (1 - alpha) * outputs[f"fragment_{key}"] + alpha * outputs[f"whole_{key}"]
        return outputs


__all__ = ["ConciseJEPASecondaryBinding"]

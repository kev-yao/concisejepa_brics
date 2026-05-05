import torch
import torch.nn as nn
import torch.nn.functional as F

from .concise import Concise


class ConciseJEPA(nn.Module):
    """
    Multi-task model:
      1) DTI classification with Concise (protein + Morgan -> [0, 1]).
      2) JEPA regression with MSE (protein + continuous FSQ drug embedding -> SMILES target embedding).

    Expected tensors:
      - protein_embedding: [B, 50, 1280]
      - morgan_fingerprint: [B, Dm]
      - smiles_target_embedding: [B, Ds]
    """

    def __init__(
        self,
        concise_backbone: Concise,
        smiles_target_dim: int = 256, # coati dimensions
        jepa_hidden_dim: int = 512,
        clip_logit_scale_init: float = 14.0,
    ) -> None:
        super().__init__()
        self.concise = concise_backbone
        self.smiles_target_dim = smiles_target_dim
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(float(clip_logit_scale_init))))

        proj_dim = self.concise.r_project.out_features
        self.jepa_predictor = nn.Sequential(
            nn.Linear(2 * proj_dim, jepa_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(jepa_hidden_dim, jepa_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(jepa_hidden_dim, smiles_target_dim),
        )

    def forward(
        self,
        protein_embedding: torch.Tensor,
        morgan_fingerprint: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        dti_out = self.concise(morgan_fingerprint, protein_embedding, is_morgan_fingerprint=True)

        context = torch.cat([dti_out["d_emb"], dti_out["r_emb"]], dim=-1)
        jepa_pred = self.jepa_predictor(context)
        drug_features = F.normalize(dti_out["d_emb"], dim=-1)
        protein_features = F.normalize(dti_out["r_emb"], dim=-1)
        similarity_logits = self.logit_scale.exp() * (drug_features @ protein_features.T)
        binding = (drug_features * protein_features).sum(dim=-1)

        outputs = {
            "binding": binding,
            "codes": dti_out["codes"],
            "jepa_pred": jepa_pred,
            "drug_features": drug_features,
            "protein_features": protein_features,
            "similarity_logits": similarity_logits,
        }

        return outputs

    def predict_from_codes(
        self,
        protein_embedding: torch.Tensor,
        drug_codes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        dti_out = self.concise(drug_codes, protein_embedding, is_morgan_fingerprint=False)

        context = torch.cat([dti_out["d_emb"], dti_out["r_emb"]], dim=-1)
        jepa_pred = self.jepa_predictor(context)
        drug_features = F.normalize(dti_out["d_emb"], dim=-1)
        protein_features = F.normalize(dti_out["r_emb"], dim=-1)
        similarity_logits = self.logit_scale.exp() * (drug_features @ protein_features.T)
        binding = (drug_features * protein_features).sum(dim=-1)

        return {
            "binding": binding,
            "codes": dti_out["codes"],
            "jepa_pred": jepa_pred,
            "drug_features": drug_features,
            "protein_features": protein_features,
            "similarity_logits": similarity_logits,
        }

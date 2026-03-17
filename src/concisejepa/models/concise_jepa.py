import torch
import torch.nn as nn

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
    ) -> None:
        super().__init__()
        self.concise = concise_backbone
        self.smiles_target_dim = smiles_target_dim

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

        outputs = {
            "binding": dti_out["binding"],
            "codes": dti_out["codes"],
            "jepa_pred": jepa_pred,
        }

        return outputs

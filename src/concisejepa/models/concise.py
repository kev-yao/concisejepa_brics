import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from .drug_decoder import DrugEncoder
from esm.model.esm2 import TransformerLayer

activation_choices = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
    "gelu": nn.GELU,
}


class Concise(nn.Module):
    """
    Use this model for reproducibility with the paper results.
    """

    def __init__(
        self,
        drug_layers,
        ligand_dim=2048,
        residue_dim=1280,
        drug_dim=128,
        proj_dim=256,
        nheads=32,
        activation="tanh",
        cosine_prediction=False,
        pairwise_attention_chunk_size=4096,
        use_pairwise_attention_checkpoint=True,
        drug_quantizer=None,
    ):
        super(Concise, self).__init__()

        self.drug_dim = len(drug_layers)
        self.pairwise_attention_chunk_size = pairwise_attention_chunk_size
        self.use_pairwise_attention_checkpoint = use_pairwise_attention_checkpoint
        self.d_encoder = DrugEncoder(
            drug_layers,
            ligand_dim,
            drug_dim,
            activation=activation_choices[activation],
            quantizer=drug_quantizer,
        )
        self.drug_token_count = self.d_encoder.num_tokens

        self.r_project = nn.Linear(residue_dim, proj_dim)
        self.d_project = nn.Linear(drug_dim, proj_dim)

        self.d_to_r_attention = nn.MultiheadAttention(proj_dim, nheads, batch_first=True)

        self.d_to_d_attention = TransformerLayer(
            embed_dim=proj_dim,
            ffn_embed_dim=proj_dim,
            attention_heads=nheads,
            use_rotary_embeddings=True,
        )

        self.r_to_r_attention = TransformerLayer(
            embed_dim=proj_dim,
            ffn_embed_dim=proj_dim,
            attention_heads=nheads,
            use_rotary_embeddings=True,
        )

        self.r_to_d_attention = nn.MultiheadAttention(proj_dim, nheads, batch_first=True)
        self.cosine_prediction = cosine_prediction

        if self.cosine_prediction:
            self.final = CosinePredictor(proj_dim * self.drug_token_count, proj_dim)
        else:
            self.final = nn.Sequential(
                nn.Linear((self.drug_token_count + 1) * proj_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, 1),
                nn.Sigmoid(),
            )

    def _pool_residue_embeddings(self, r_emb):
        r_emb_wt = F.softmax(10 * r_emb, dim=1)
        return (r_emb * r_emb_wt).sum(dim=1)

    def _score_aligned_embeddings(self, d_pair, r_pair):
        d_pair = rearrange(d_pair, "b n k -> b (n k)")
        if self.cosine_prediction:
            return self.final(d_pair, r_pair)

        final_input = torch.cat([d_pair, r_pair], dim=-1)
        return self.final(final_input).squeeze(-1)

    def _attend_and_score_pair_chunk(self, d_pair, r_pair, r_pooled_pair):
        d_r_mixed, _ = self.r_to_d_attention(d_pair, r_pair, r_pair, need_weights=False)
        d_pair = d_pair + d_r_mixed
        scores = self._score_aligned_embeddings(d_pair, r_pooled_pair)
        return scores, d_pair

    def _pairwise_drug_to_protein_scores(self, d_emb, r_emb, r_pooled):
        batch_size, drug_tokens, proj_dim = d_emb.shape
        total_pairs = batch_size * batch_size
        chunk_size = min(int(self.pairwise_attention_chunk_size), total_pairs)
        pairwise_binding = d_emb.new_empty(batch_size, batch_size)
        diagonal_d_emb = d_emb.new_empty(batch_size, drug_tokens, proj_dim)

        for start in range(0, total_pairs, chunk_size):
            end = min(start + chunk_size, total_pairs)
            pair_indices = torch.arange(start, end, device=d_emb.device)
            drug_indices = torch.div(pair_indices, batch_size, rounding_mode="floor")
            protein_indices = pair_indices.remainder(batch_size)

            d_pair = d_emb.index_select(0, drug_indices)
            r_pair = r_emb.index_select(0, protein_indices)
            r_pooled_pair = r_pooled.index_select(0, protein_indices)
            if self.use_pairwise_attention_checkpoint and torch.is_grad_enabled():
                scores, d_pair = checkpoint(
                    self._attend_and_score_pair_chunk,
                    d_pair,
                    r_pair,
                    r_pooled_pair,
                    use_reentrant=False,
                )
            else:
                scores, d_pair = self._attend_and_score_pair_chunk(d_pair, r_pair, r_pooled_pair)

            pairwise_binding.reshape(-1)[start:end] = scores

            diagonal_mask = drug_indices == protein_indices
            if diagonal_mask.any():
                diagonal_d_emb.index_copy_(0, drug_indices[diagonal_mask], d_pair[diagonal_mask])

        return pairwise_binding, diagonal_d_emb

    def emb(self, x):
        d_outs = self.d_encoder(x)
        d_emb = d_outs["emb"]
        d_emb = self.d_project(d_emb)
        d_d_mixed, _ = self.d_to_d_attention(rearrange(d_emb, "b n k -> n b k"))
        d_d_mixed = rearrange(d_d_mixed, "n b k -> b n k")
        d_emb = d_emb + d_d_mixed
        return d_emb

    def prot_emb(self, x):
        assert len(x.shape) == 3, "Input must be of shape [batch, 50, 1280]"
        r_emb = self.r_project(x)
        r_r_mixed, _ = self.r_to_r_attention(rearrange(r_emb, "b n k -> n b k"))
        r_r_mixed = rearrange(r_r_mixed, "n b k -> b n k")
        r_emb = r_emb + r_r_mixed
        return r_emb.mean(dim=1)

    def codes(self, x):
        d_outs = self.d_encoder(x)
        return d_outs["codes"]

    def score(self, r_emb, b_size=32):
        device = next(self.parameters()).device
        combinations = self.d_encoder.all_code_indices(device=device)
        scores = []
        with torch.no_grad():
            for chunk in combinations.split(b_size):
                r_emb_p = r_emb.repeat(chunk.shape[0], 1, 1)
                scores.append(self.forward(chunk, r_emb_p, is_morgan_fingerprint=False)["binding"])
        scores = torch.cat(scores).cpu()
        scores, idxs = torch.sort(scores, descending=True)
        return scores, combinations[idxs]

    def forward(
        self,
        d_emb,
        r_emb,
        is_morgan_fingerprint=True,
    ):
        # encode drug
        if is_morgan_fingerprint:
            d_outs = self.d_encoder(d_emb)
            d_emb, d_codes = (d_outs["emb"], d_outs["codes"])
        else:
            d_codes = d_emb
            d_emb = self.d_encoder.embed(d_emb)

        d_emb = self.d_project(d_emb)
        r_emb = self.r_project(r_emb)

        d_d_mixed, _ = self.d_to_d_attention(rearrange(d_emb, "b n k -> n b k"))
        d_d_mixed = rearrange(d_d_mixed, "n b k -> b n k")
        d_emb = d_emb + d_d_mixed
        r_r_mixed, _ = self.r_to_r_attention(rearrange(r_emb, "b n k -> n b k"))
        r_r_mixed = rearrange(r_r_mixed, "n b k -> b n k")
        r_emb = r_emb + r_r_mixed

        # do a softmax reduction of the residues into a single column
        r_pooled = self._pool_residue_embeddings(r_emb)
        pairwise_binding, d_emb = self._pairwise_drug_to_protein_scores(d_emb, r_emb, r_pooled)
        binding = pairwise_binding.diagonal()

        # Use the matched drug-protein pairs for per-row features consumed by JEPA.
        d_emb = rearrange(d_emb, "b n k -> b (n k)")
        r_emb = r_pooled

        result = {
            "d_emb": d_emb,
            "r_emb": r_emb,
            "binding": binding,
            "pairwise_binding": pairwise_binding,
            "codes": d_codes,
        }
        if is_morgan_fingerprint:
            result["pre_quantized"] = d_outs["pre_quantized"]
            result["quantized"] = d_outs["quantized"]
        return result


class ConciseSA(nn.Module):
    """
    Same Concise but with self attention added on the fixed-length receptor embeddings
    output from Raygun. Use the other model to reproduce the paper results.
    """

    def __init__(
        self,
        drug_layers,
        ligand_dim=2048,
        residue_dim=1280,
        drug_dim=128,
        proj_dim=256,
        nheads=32,
        activation="tanh",
        cosine_prediction=False,
        drug_quantizer=None,
    ):
        super(ConciseSA, self).__init__()

        self.drug_dim = len(drug_layers)
        self.d_encoder = DrugEncoder(
            drug_layers,
            ligand_dim,
            drug_dim,
            activation=activation_choices[activation],
            quantizer=drug_quantizer,
        )
        self.drug_token_count = self.d_encoder.num_tokens

        self.r_project = nn.Linear(residue_dim, proj_dim)
        self.d_project = nn.Linear(drug_dim, proj_dim)

        self.d_to_d_attention = TransformerLayer(
            embed_dim=proj_dim,
            ffn_embed_dim=proj_dim,
            attention_heads=nheads,
            use_rotary_embeddings=True,
        )

        self.r_to_r_attention = TransformerLayer(
            embed_dim=proj_dim,
            ffn_embed_dim=proj_dim,
            attention_heads=nheads,
            use_rotary_embeddings=True,
        )

        self.d_to_r_attention = nn.MultiheadAttention(proj_dim, nheads, batch_first=True)

        self.r_to_d_attention = nn.MultiheadAttention(proj_dim, nheads, batch_first=True)

        self.d_to_d_attention_2 = TransformerLayer(
            embed_dim=proj_dim,
            ffn_embed_dim=proj_dim,
            attention_heads=nheads,
            use_rotary_embeddings=True,
        )

        self.r_to_r_attention_2 = TransformerLayer(
            embed_dim=proj_dim,
            ffn_embed_dim=proj_dim,
            attention_heads=nheads,
            use_rotary_embeddings=True,
        )

        self.cosine_prediction = cosine_prediction

        if self.cosine_prediction:
            self.final = CosinePredictor(proj_dim * self.drug_token_count, proj_dim)
        else:
            self.final = nn.Sequential(
                nn.Linear((self.drug_token_count + 1) * proj_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, 1),
                nn.Sigmoid(),
            )

    def emb(self, x):
        d_outs = self.d_encoder(x)
        d_emb = d_outs["emb"]
        d_emb = self.d_project(d_emb)
        d_d_mixed, _ = self.d_to_d_attention(rearrange(d_emb, "b n k -> n b k"))
        d_d_mixed = rearrange(d_d_mixed, "n b k -> b n k")
        d_emb = d_emb + d_d_mixed
        return d_emb

    def prot_emb(self, x):
        assert len(x.shape) == 3, "Input must be of shape [batch, 50, 1280]"
        r_emb = self.r_project(x)
        r_r_mixed, _ = self.r_to_r_attention(rearrange(r_emb, "b n k -> n b k"))
        r_r_mixed = rearrange(r_r_mixed, "n b k -> b n k")
        r_emb = r_emb + r_r_mixed
        return r_emb.mean(dim=1)

    def codes(self, x):
        d_outs = self.d_encoder(x)
        return d_outs["codes"]

    def score(self, r_emb, b_size=32, cutoff=0.5):
        device = next(self.parameters()).device
        with torch.no_grad():
            r_emb = r_emb.to(device)
            r_emb_proj = self.r_project(r_emb)
            r_r_mixed, _ = self.r_to_r_attention(rearrange(r_emb_proj, "b n k -> n b k"))
            r_emb_processed = rearrange(r_r_mixed, "n b k -> b n k") + r_emb_proj

        combinations = self.d_encoder.all_code_indices(device=device)
        scores = []
        with torch.no_grad():
            for chunk in combinations.split(b_size):
                r_emb_proj = r_emb_processed.expand(chunk.size(0), -1, -1)
                scores.append(self.forward_cached(chunk, r_emb_proj, None, is_morgan_fingerprint=False)["binding"])
        scores = torch.cat(scores)
        mask = scores >= cutoff
        scores = scores[mask]
        combinations = combinations[mask]
        scores, idxs = torch.sort(scores, descending=True)
        return scores.cpu(), combinations[idxs].cpu()

    def forward(self, d_emb, r_emb, is_morgan_fingerprint=True):
        # encode drug
        if is_morgan_fingerprint:
            d_outs = self.d_encoder(d_emb)
            d_emb, d_codes = (d_outs["emb"], d_outs["codes"])
        else:
            d_codes = d_emb
            d_emb = self.d_encoder.embed(d_emb)

        d_emb = self.d_project(d_emb)
        r_emb = self.r_project(r_emb)

        d_d_mixed, _ = self.d_to_d_attention(rearrange(d_emb, "b n k -> n b k"))
        d_d_mixed = rearrange(d_d_mixed, "n b k -> b n k")
        d_emb = d_emb + d_d_mixed
        r_r_mixed, _ = self.r_to_r_attention(rearrange(r_emb, "b n k -> n b k"))
        r_r_mixed = rearrange(r_r_mixed, "n b k -> b n k")
        r_emb = r_emb + r_r_mixed

        d_r_mixed, _ = self.r_to_d_attention(d_emb, r_emb, r_emb)
        d_emb = d_emb + d_r_mixed

        r_d_mixed, _ = self.d_to_r_attention(r_emb, d_emb, d_emb)
        r_emb = r_emb + r_d_mixed

        d_d_mixed, _ = self.d_to_d_attention_2(rearrange(d_emb, "b n k -> n b k"))
        d_d_mixed = rearrange(d_d_mixed, "n b k -> b n k")
        d_emb = d_emb + d_d_mixed

        r_r_mixed, _ = self.r_to_r_attention_2(rearrange(r_emb, "b n k -> n b k"))
        r_r_mixed = rearrange(r_r_mixed, "n b k -> b n k")
        r_emb = r_emb + r_r_mixed

        # do a softmax reduction of the residues into a single column
        r_emb_wt = F.softmax(10 * r_emb, dim=1)
        r_emb = (r_emb * r_emb_wt).sum(dim=1)
        d_emb = rearrange(d_emb, "b n k -> b (n k)")

        if self.cosine_prediction:
            return {
                "binding": self.final(d_emb, r_emb).squeeze(-1),
                "codes": d_codes,
            }

        return {
            "binding": self.final(torch.cat([d_emb, r_emb], dim=-1)).squeeze(-1),
            "codes": d_codes,
        }

    def forward_cached(self, d_emb, r_emb, r_emb_processed, is_morgan_fingerprint=True):
        """
        Args:
            d_emb: Drug embeddings
            r_emb_processed: Already processed receptor embeddings (after projection and self-attention)
            is_morgan_fingerprint: Whether input is Morgan fingerprint
        """
        with torch.no_grad():
            # encode drug
            if is_morgan_fingerprint:
                d_outs = self.d_encoder(d_emb)
                d_emb, d_codes = (d_outs["emb"], d_outs["codes"])
            else:
                d_codes = d_emb
                d_emb = self.d_encoder.embed(d_emb)

            d_emb = self.d_project(d_emb)

            # Drug self-attention
            d_d_mixed, _ = self.d_to_d_attention(rearrange(d_emb, "b n k -> n b k"))
            d_emb = d_emb + rearrange(d_d_mixed, "n b k -> b n k")

            # Cross attention with pre-processed receptor embeddings
            d_r_mixed, _ = self.r_to_d_attention(d_emb, r_emb, r_emb)
            d_emb = d_emb + d_r_mixed

            r_d_mixed, _ = self.d_to_r_attention(r_emb, d_emb, d_emb)
            r_emb = r_emb + r_d_mixed

            d_d_mixed, _ = self.d_to_d_attention_2(rearrange(d_emb, "b n k -> n b k"))
            d_d_mixed = rearrange(d_d_mixed, "n b k -> b n k")
            d_emb = d_emb + d_d_mixed

            r_r_mixed, _ = self.r_to_r_attention_2(rearrange(r_emb, "b n k -> n b k"))
            r_r_mixed = rearrange(r_r_mixed, "n b k -> b n k")
            r_emb = r_emb + r_r_mixed

            # Softmax reduction
            r_emb_wt = F.softmax(10 * r_emb, dim=1)
            r_emb = (r_emb * r_emb_wt).sum(dim=1)

            # mean over the middle dimension of d_emb
            d_emb = rearrange(d_emb, "b n k -> b (n k)")
            if self.cosine_prediction:
                return {
                    "binding": self.final(d_emb, r_emb),
                    "codes": d_codes,
                }
            return {
                "binding": self.final(torch.cat([d_emb, r_emb], dim=-1)).squeeze(-1),
                "codes": d_codes,
            }


class CosinePredictor(nn.Module):
    def __init__(self, drug_dim, proj_dim):
        super().__init__()
        self.ligand = nn.Sequential(nn.Linear(drug_dim, proj_dim), nn.ReLU())
        self.rec = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.ReLU())
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(self, ligand, rec):
        ligand = self.ligand(ligand)
        rec = self.rec(rec)
        return self.cosine(ligand, rec)


def load_concise(model, path):
    model.load_state_dict(torch.load(path)["model_state_dict"])
    return model

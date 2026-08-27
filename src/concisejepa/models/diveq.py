import torch
from torch import nn

from .dynamic_tanh import DynamicTanh


class DiVeQQuantizer(nn.Module):
    """Learned nearest-neighbor vector quantizer with DiVeQ gradients."""

    def __init__(
        self,
        codebook_size: int = 1024,
        code_dim: int = 128,
        sigma: float = 0.01,
        replacement_interval: int = 100,
        discard_threshold: float = 0.01,
        replacement_perturbation: float = 1e-9,
    ) -> None:
        super().__init__()
        if codebook_size <= 0 or code_dim <= 0:
            raise ValueError("codebook_size and code_dim must be positive")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        if replacement_interval <= 0:
            raise ValueError("replacement_interval must be positive")
        if not 0 <= discard_threshold <= 1:
            raise ValueError("discard_threshold must be in [0, 1]")
        if replacement_perturbation < 0:
            raise ValueError("replacement_perturbation must be non-negative")
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self.sigma = float(sigma)
        self.replacement_interval = int(replacement_interval)
        self.discard_threshold = float(discard_threshold)
        self.replacement_perturbation = float(replacement_perturbation)

        codebook = torch.rand(self.codebook_size, self.code_dim) / self.codebook_size
        self.codebook = nn.Parameter(codebook)
        self.register_buffer("assignment_counts", torch.zeros(self.codebook_size, dtype=torch.long))
        self.register_buffer("training_steps", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[-1] != self.code_dim:
            raise ValueError(f"Expected input shape [B, {self.code_dim}], got {tuple(x.shape)}")
        if self.training and int(self.training_steps.item()) > 0:
            if int(self.training_steps.item()) % self.replacement_interval == 0:
                self._replace_inactive_codes()

        indices = torch.cdist(x, self.codebook).argmin(dim=-1)
        selected = self.embed(indices)
        if not self.training:
            return selected, indices

        direction = selected - x
        error_magnitude = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        noisy_direction = direction + self.sigma * torch.randn_like(direction)
        normalized_direction = noisy_direction / torch.linalg.vector_norm(
            noisy_direction, dim=-1, keepdim=True
        ).clamp_min(1e-12)
        quantized = x + error_magnitude * normalized_direction.detach()

        with torch.no_grad():
            self.assignment_counts += torch.bincount(indices, minlength=self.codebook_size)
            self.training_steps += 1
        return quantized, indices

    @torch.no_grad()
    def _replace_inactive_codes(self) -> None:
        usage_per_step = self.assignment_counts.float() / self.replacement_interval
        inactive = torch.where(usage_per_step < self.discard_threshold)[0]
        active = torch.where(usage_per_step >= self.discard_threshold)[0]
        if inactive.numel() > 0 and active.numel() > 0:
            probabilities = self.assignment_counts[active].float()
            probabilities /= probabilities.sum()
            sampled = active[torch.multinomial(probabilities, inactive.numel(), replacement=True)]
            replacements = self.codebook[sampled].clone()
            replacements += self.replacement_perturbation * torch.randn_like(replacements)
            self.codebook[inactive] = replacements
        self.assignment_counts.zero_()

    def embed(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(device=self.codebook.device, dtype=torch.long)
        if torch.any(indices < 0) or torch.any(indices >= self.codebook_size):
            raise IndexError(f"DiVeQ indices must be in [0, {self.codebook_size - 1}]")
        return self.codebook[indices]


class ProductDiVeQQuantizer(nn.Module):
    """Product DiVeQ with independent learned codebooks and one reconstructed token."""

    def __init__(
        self,
        dim: int = 128,
        num_groups: int = 3,
        codebook_size: int = 32,
        group_dim: int = 1,
        sigma: float = 0.01,
        replacement_interval: int = 100,
        discard_threshold: float = 0.01,
        replacement_perturbation: float = 1e-9,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        if num_groups <= 0 or group_dim <= 0:
            raise ValueError("num_groups and group_dim must be positive")

        self.dim = int(dim)
        self.num_groups = int(num_groups)
        self.codebook_size = int(codebook_size)
        self.group_dim = int(group_dim)
        self.product_dim = self.num_groups * self.group_dim
        self.total_capacity = self.codebook_size**self.num_groups
        self.sigma = float(sigma)
        self.replacement_interval = int(replacement_interval)
        self.discard_threshold = float(discard_threshold)

        scale = 2 if activation is nn.Tanh else 1
        self.ln1 = DynamicTanh(self.dim, channels_last=True)
        self.in_proj = nn.Sequential(
            nn.Linear(self.dim, scale * self.dim),
            nn.GELU(),
            nn.Linear(scale * self.dim, self.product_dim),
        )
        self.quantizers = nn.ModuleList(
            [
                DiVeQQuantizer(
                    codebook_size=self.codebook_size,
                    code_dim=self.group_dim,
                    sigma=sigma,
                    replacement_interval=replacement_interval,
                    discard_threshold=discard_threshold,
                    replacement_perturbation=replacement_perturbation,
                )
                for _ in range(self.num_groups)
            ]
        )
        self.out_proj = nn.Sequential(
            nn.Linear(self.product_dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self.ln2 = nn.LayerNorm(self.dim)
        self.activation = activation()

    def _reconstruct(self, quantized_groups: torch.Tensor) -> torch.Tensor:
        batch_size = quantized_groups.shape[0]
        flat = quantized_groups.reshape(batch_size, 1, self.product_dim)
        return self.activation(self.ln2(self.out_proj(flat)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1:] != (1, self.dim):
            raise ValueError(f"Expected input shape [B, 1, {self.dim}], got {tuple(x.shape)}")
        projected = self.in_proj(self.ln1(x)).reshape(x.shape[0], self.num_groups, self.group_dim)
        selected_groups = []
        indices = []
        for group, quantizer in enumerate(self.quantizers):
            if self.training and int(quantizer.training_steps.item()) > 0:
                if int(quantizer.training_steps.item()) % quantizer.replacement_interval == 0:
                    quantizer._replace_inactive_codes()
            group_indices = torch.cdist(projected[:, group], quantizer.codebook).argmin(dim=-1)
            selected_groups.append(quantizer.embed(group_indices))
            indices.append(group_indices)
            if self.training:
                with torch.no_grad():
                    quantizer.assignment_counts += torch.bincount(
                        group_indices,
                        minlength=quantizer.codebook_size,
                    )
                    quantizer.training_steps += 1

        selected_groups = torch.stack(selected_groups, dim=1)
        codes = torch.stack(indices, dim=1)
        if self.training:
            projected_flat = projected.reshape(x.shape[0], self.product_dim)
            selected_flat = selected_groups.reshape(x.shape[0], self.product_dim)
            direction = selected_flat - projected_flat
            error_magnitude = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
            noisy_direction = direction + self.sigma * torch.randn_like(direction)
            normalized_direction = noisy_direction / torch.linalg.vector_norm(
                noisy_direction,
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
            quantized_groups = (
                projected_flat + error_magnitude * normalized_direction.detach()
            ).reshape(x.shape[0], self.num_groups, self.group_dim)
        else:
            quantized_groups = selected_groups
        return {
            "codes": codes,
            "emb": self._reconstruct(quantized_groups),
            "points": quantized_groups,
            "pre_quantized": projected,
            "quantized": quantized_groups,
        }

    def embed(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 2 or codes.shape[1] != self.num_groups:
            raise ValueError(f"Expected codes shape [B, {self.num_groups}], got {tuple(codes.shape)}")
        quantized_groups = torch.stack(
            [quantizer.embed(codes[:, group]) for group, quantizer in enumerate(self.quantizers)],
            dim=1,
        )
        return self._reconstruct(quantized_groups)

    def all_code_indices(self, device: torch.device | None = None) -> torch.Tensor:
        axes = [torch.arange(self.codebook_size, device=device) for _ in range(self.num_groups)]
        return torch.cartesian_prod(*axes)

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 2 or codes.shape[1] != self.num_groups:
            raise ValueError(f"Expected codes shape [B, {self.num_groups}], got {tuple(codes.shape)}")
        basis = self.codebook_size ** torch.arange(self.num_groups, device=codes.device, dtype=torch.long)
        return (codes.to(torch.long) * basis.unsqueeze(0)).sum(dim=1)

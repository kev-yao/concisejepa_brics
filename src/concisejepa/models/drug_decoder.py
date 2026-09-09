import torch
from einops.layers.torch import Rearrange
import torch.nn as nn
from typing import Any, List, Dict, Type
from .fsq import ResidualFSQ
from .diveq import ProductDiVeQQuantizer


class DrugEncoder(torch.nn.Module):
    """

    Drug encoder with residual finite scalar quantization layers.

    ``quantizer={"type": "continuous_fsq"}`` is a matched no-rounding control:
    it retains the FSQ projections, bounds, and scale, but returns continuous
    points and -1 code sentinels. It has no discrete embedding/enumeration API.
    Keep the quantizer type in checkpoint configuration: parameter keys/shapes
    intentionally match FSQ, so a state dict alone does not identify the mode.

    Args:
        layers (List[List[int]]): Quantization levels for each ResidualFSQ layer.
        dim (int, optional): Input dimension of the drug representation. Defaults to 2048.
        latent_dim (int, optional): Dimension of the latent space after pre-transformation.
            Defaults to 128.
        activation (Type[torch.nn.Module], optional): Activation function to use throughout
            the network. Defaults to nn.Tanh.

    Attributes:
        residualfsqs (torch.nn.ModuleList): ResidualFSQ layers.
        pre_transform (torch.nn.Sequential): Fingerprint projection before quantization.

    Shape:
        - Input: (batch_size, dim)
        - Output: Dict containing:
            - codes: (batch_size, num_layers)
            - emb: (batch_size, num_layers, latent_dim)
            - points: (batch_size, num_layers, fsq_dim)
    """

    residualfsqs: torch.nn.ModuleList
    pre_transform: torch.nn.Sequential

    def __init__(
        self,
        layers: List[List[int]],
        dim: int = 2048,
        latent_dim: int = 128,
        activation: Type[torch.nn.Module] = nn.Tanh,
        quantizer: Dict[str, Any] | None = None,
    ) -> None:
        super(DrugEncoder, self).__init__()

        # Initialize pre-transformation network
        self.pre_transform = self._build_pre_transform(dim, latent_dim, activation)

        quantizer = dict(quantizer or {"type": "fsq"})
        self.quantizer_type = str(quantizer.get("type", "fsq")).lower()
        self.residualfsqs = torch.nn.ModuleList()
        self.diveq = None
        if self.quantizer_type in ("fsq", "continuous_fsq"):
            if len(layers) != 1:
                raise ValueError("Hybrid FSQ requires exactly one drug_layers entry")
            self.residualfsqs = self._build_residual_fsq_layers(
                layers, latent_dim, activation, discretize=self.quantizer_type == "fsq"
            )
            if not self.residualfsqs:
                raise ValueError("FSQ requires at least one drug_layers entry")
            self.num_tokens = len(self.residualfsqs)
            self.codebook_size = None
        elif self.quantizer_type == "diveq":
            num_groups = int(quantizer.get("num_groups", 3))
            codebook_size = int(quantizer.get("codebook_size", 32))
            group_dim = int(quantizer.get("group_dim", 1))
            expected_levels = [codebook_size] * num_groups
            if len(layers) != 1 or list(layers[0]) != expected_levels:
                raise ValueError(
                    "Product DiVeQ capacity must match drug_layers; expected "
                    f"drug_layers=[{expected_levels}], got {layers}"
                )
            self.diveq = ProductDiVeQQuantizer(
                dim=latent_dim,
                num_groups=num_groups,
                codebook_size=codebook_size,
                group_dim=group_dim,
                sigma=float(quantizer.get("sigma", 0.01)),
                replacement_interval=int(quantizer.get("replacement_interval", 100)),
                discard_threshold=float(quantizer.get("discard_threshold", 0.01)),
                replacement_perturbation=float(quantizer.get("replacement_perturbation", 1e-9)),
                activation=activation,
            )
            self.num_tokens = 1
            self.codebook_size = self.diveq.codebook_size
            self.num_groups = self.diveq.num_groups
            self.total_code_capacity = self.diveq.total_capacity
        else:
            raise ValueError(f"Unknown drug quantizer type: {self.quantizer_type!r}")

    def _build_pre_transform(self, dim: int, latent_dim: int, activation: Type[torch.nn.Module]) -> torch.nn.Sequential:
        """Builds the pre-transformation network."""
        return torch.nn.Sequential(
            Rearrange("b d -> b 1 d"),  # Add sequence dimension
            torch.nn.Linear(dim, dim // 2),
            activation(),
            torch.nn.Linear(dim // 2, latent_dim),
            torch.nn.Identity(),
        )

    def _build_residual_fsq_layers(
        self,
        layers: List[List[int]],
        latent_dim: int,
        activation: Type[torch.nn.Module],
        discretize: bool = True,
    ) -> torch.nn.ModuleList:
        return torch.nn.ModuleList(
            [
                ResidualFSQ(
                    layer_config,
                    idx,
                    dim=latent_dim,
                    activation=activation,
                    discretize=discretize,
                )
                for idx, layer_config in enumerate(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Project and quantize Morgan fingerprints.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - codes: Quantization indices
                - emb: Quantized embeddings
                - points: Quantized points before output projection
        """
        x = self.pre_transform(x)

        if self.quantizer_type == "diveq":
            return self.diveq(x)

        res = self.residualfsqs[0](x, return_code=True)
        quantized = res["quantized"]
        points = res["points"]
        if self.quantizer_type == "continuous_fsq":
            factors = torch.full_like(points.squeeze(1), -1, dtype=torch.long)
        else:
            factors = self.residualfsqs[0].fsq.codes_to_factors(points).squeeze(1)
        return {
            "codes": factors,
            "emb": quantized,
            "points": points,
            "pre_quantized": x,
            "quantized": quantized,
        }

    def embed(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Convert quantization codes back to embeddings.

        Args:
            codes (torch.Tensor): Factor indices of shape (batch_size, codebook_dim)

        Returns:
            torch.Tensor: Reconstructed embeddings of shape (batch_size, 1, latent_dim)
        """
        if self.quantizer_type == "continuous_fsq":
            raise RuntimeError("continuous_fsq has no discrete codes to embed")
        if self.quantizer_type == "diveq":
            return self.diveq.embed(codes)

        rfsq = self.residualfsqs[0]
        points = rfsq.fsq.factors_to_codes(codes.to(torch.long)).unsqueeze(1)
        return rfsq.activation(rfsq.ln2(rfsq.out_proj(points)))

    def set_levels(self, levels: List[int], device: torch.device) -> None:
        """
        Set quantization levels for all ResidualFSQ layers.

        Args:
            levels (List[int]): List of quantization levels to set
            device (torch.device): Device to place the new levels on
        """
        if self.quantizer_type != "fsq":
            raise RuntimeError("set_levels is only available for the FSQ quantizer")
        for rfsq in self.residualfsqs:
            rfsq.set_levels(levels, device)

    def get_levels(self) -> List[List[int]]:
        """
        Get quantization (or continuous-control bounding) levels from each layer.

        Returns:
            List[List[int]]: List of quantization levels for each layer
        """
        if self.quantizer_type not in ("fsq", "continuous_fsq"):
            raise RuntimeError("get_levels is only available for FSQ-based encoders")
        return [rfsq.fsq._levels for rfsq in self.residualfsqs]

    def all_code_indices(self, device: torch.device | None = None) -> torch.Tensor:
        if self.quantizer_type == "continuous_fsq":
            raise RuntimeError("continuous_fsq has no discrete codebook to enumerate")
        if self.quantizer_type == "diveq":
            return self.diveq.all_code_indices(device=device)
        levels = self.get_levels()[0]
        axes = [torch.arange(int(level), device=device) for level in levels]
        return torch.cartesian_prod(*axes)

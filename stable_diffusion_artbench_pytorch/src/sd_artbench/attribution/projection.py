"""Memory-bounded deterministic projections for LoRA gradient trees."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFFFFFF


def tree_l2_norm(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    total = None
    for tensor in tensors:
        value = tensor.detach().float().square().sum()
        total = value if total is None else total + value
    if total is None:
        raise ValueError("cannot compute a norm of an empty gradient tree")
    return total.sqrt()


@dataclass(frozen=True)
class CountSketchProjector:
    """Project tensors without materializing a ``parameter_count x proj_dim`` matrix.

    Hashes are generated from global flattened coordinates, so query, raw-gradient,
    and AdamW features share exactly the same checkpoint-specific projection.
    """

    proj_dim: int
    seed: int
    chunk_size: int = 1_000_000

    def __post_init__(self) -> None:
        if self.proj_dim <= 0:
            raise ValueError("proj_dim must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    def project(
        self,
        tensors: Sequence[torch.Tensor],
        *,
        full_l2_normalize: bool = False,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, float]:
        if not tensors:
            raise ValueError("cannot project an empty gradient tree")
        device = tensors[0].device
        norm_tensor = tree_l2_norm(tensors)
        norm = float(norm_tensor.detach().cpu())
        scale = 1.0 / max(norm, eps) if full_l2_normalize else 1.0
        output = torch.zeros(self.proj_dim, dtype=torch.float32, device=device)

        # Constants are kept below signed int64 overflow limits.  The arithmetic
        # intentionally wraps in int64 and remains deterministic on CPU/CUDA.
        bucket_a = 6364136223846793005
        bucket_b = (1442695040888963407 + self.seed * 2_654_435_761) & 0x7FFFFFFFFFFFFFFF
        sign_a = 2862933555777941757
        sign_b = (3037000493 + self.seed * 1_000_003) & 0x7FFFFFFFFFFFFFFF
        offset = 0
        for tensor in tensors:
            flat = tensor.detach().float().reshape(-1)
            for start in range(0, flat.numel(), self.chunk_size):
                end = min(start + self.chunk_size, flat.numel())
                coordinates = torch.arange(
                    offset + start,
                    offset + end,
                    dtype=torch.int64,
                    device=device,
                )
                buckets = torch.remainder(
                    coordinates * bucket_a + bucket_b, self.proj_dim
                )
                sign_bits = torch.bitwise_and(coordinates * sign_a + sign_b, 1)
                signs = sign_bits.to(torch.float32).mul_(2).sub_(1)
                output.scatter_add_(0, buckets, flat[start:end] * signs * scale)
            offset += flat.numel()
        return output, norm

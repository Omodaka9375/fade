"""Phase 3: product-quantization codebook for the deepest compression tier.

Per-layer, per-head product quantization of K and V vectors:
    - Train codebook via k-means over calibration K/V activations.
    - At inference, store 1 uint8 code per sub-vector; decode via table lookup.
    - ``sub_dim=32``, ``K=256`` gives ~2 effective bits per element.

Requires ``scikit-learn`` for training (``pip install fade[codebook]``).
Encode/decode are pure-torch and run on any device.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# --- knobs ------------------------------------------------------------------- #
DEFAULT_SUB_DIM: int = 32
DEFAULT_NUM_CENTROIDS: int = 256


@dataclass
class PQCodebook:
    """Product-quantization codebook for a single (layer, head) pair.

    Attributes:
        centroids: [n_sub, num_centroids, sub_dim] float tensor of centroids.
        sub_dim: size of each sub-vector.
    """

    centroids: Tensor
    sub_dim: int

    @property
    def n_sub(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def num_centroids(self) -> int:
        return int(self.centroids.shape[1])

    @property
    def head_dim(self) -> int:
        return self.n_sub * self.sub_dim

    @classmethod
    def train(
        cls,
        vectors: Tensor,
        sub_dim: int = DEFAULT_SUB_DIM,
        num_centroids: int = DEFAULT_NUM_CENTROIDS,
    ) -> PQCodebook:
        """Train a PQ codebook from calibration vectors.

        Args:
            vectors: [N, head_dim] float activations from a calibration run.
            sub_dim: sub-vector length (must divide head_dim).
            num_centroids: entries per sub-codebook (max 256 for uint8 codes).

        Returns:
            A trained ``PQCodebook`` with centroids on CPU.
        """
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError as e:
            raise ImportError(
                "PQCodebook.train requires scikit-learn. Install with: pip install fade[codebook]"
            ) from e

        N, D = vectors.shape
        if D % sub_dim != 0:
            raise ValueError(f"head_dim={D} must be divisible by sub_dim={sub_dim}")
        if num_centroids > 256:
            raise ValueError(f"num_centroids={num_centroids} exceeds uint8 max (256)")

        n_sub = D // sub_dim
        # [N, n_sub, sub_dim]
        subs = vectors.float().cpu().numpy().reshape(N, n_sub, sub_dim)

        centroids_list = []
        for s in range(n_sub):
            km = MiniBatchKMeans(
                n_clusters=num_centroids,
                batch_size=min(1024, N),
                max_iter=50,
                n_init=1,
                random_state=42,
            )
            km.fit(subs[:, s, :])  # [N, sub_dim]
            centroids_list.append(torch.from_numpy(km.cluster_centers_).float())

        # [n_sub, num_centroids, sub_dim]
        centroids = torch.stack(centroids_list, dim=0)
        return cls(centroids=centroids, sub_dim=sub_dim)

    def encode(self, vectors: Tensor) -> Tensor:
        """Encode [..., head_dim] vectors to [..., n_sub] uint8 codes.

        Uses nearest-centroid assignment per sub-vector slot.
        """
        orig_shape = vectors.shape
        D = orig_shape[-1]
        if self.head_dim != D:
            raise ValueError(f"last dim {D} != codebook head_dim {self.head_dim}")

        flat = vectors.reshape(-1, D).float()  # [M, D]
        M = flat.shape[0]
        subs = flat.view(M, self.n_sub, self.sub_dim)  # [M, n_sub, sub_dim]

        centroids = self.centroids.to(flat.device)  # [n_sub, K, sub_dim]
        codes = torch.empty(M, self.n_sub, dtype=torch.uint8, device=flat.device)
        for s in range(self.n_sub):
            # [M, sub_dim] vs [K, sub_dim] -> [M, K]
            dists = torch.cdist(subs[:, s, :], centroids[s])  # [M, K]
            codes[:, s] = dists.argmin(dim=-1).to(torch.uint8)

        return codes.view(*orig_shape[:-1], self.n_sub)

    def decode(self, codes: Tensor) -> Tensor:
        """Decode [..., n_sub] uint8 codes back to [..., head_dim] floats.

        Gathers centroids per sub-vector slot and concatenates.
        """
        orig_shape = codes.shape
        flat = codes.reshape(-1, self.n_sub).long()  # [M, n_sub]
        flat.shape[0]

        centroids = self.centroids.to(flat.device)  # [n_sub, K, sub_dim]
        parts = []
        for s in range(self.n_sub):
            idx = flat[:, s]  # [M]
            # Gather: centroids[s][idx] -> [M, sub_dim]
            parts.append(centroids[s].index_select(0, idx))

        # [M, head_dim]
        decoded = torch.cat(parts, dim=-1)
        return decoded.view(*orig_shape[:-1], self.head_dim)


@dataclass
class SharedPQCodebook:
    """Cross-layer shared PQ codebook.

    Instead of one codebook per (layer, head), shares a single codebook
    across a group of adjacent layers. Reduces codebook memory ~Nx where
    N is the group size, at a small quality cost.

    Usage:
        shared = SharedPQCodebook.train(vectors_from_layers, group_size=4, ...)
        # shared.codebook is a single PQCodebook used for all layers in the group
    """

    codebook: PQCodebook
    group_size: int
    layer_range: tuple[int, int]  # (start_layer, end_layer) inclusive

    @classmethod
    def train(
        cls,
        vectors_per_layer: list[Tensor],
        group_size: int = 4,
        sub_dim: int = DEFAULT_SUB_DIM,
        num_centroids: int = DEFAULT_NUM_CENTROIDS,
    ) -> list[SharedPQCodebook]:
        """Train shared codebooks from per-layer calibration vectors.

        Args:
            vectors_per_layer: list of [N_i, head_dim] tensors, one per layer.
            group_size: number of adjacent layers sharing one codebook.

        Returns:
            List of SharedPQCodebook, one per group.
        """
        n_layers = len(vectors_per_layer)
        groups = []
        for start in range(0, n_layers, group_size):
            end = min(start + group_size, n_layers)
            # Pool vectors from all layers in the group.
            pooled = torch.cat(vectors_per_layer[start:end], dim=0)
            cb = PQCodebook.train(pooled, sub_dim=sub_dim, num_centroids=num_centroids)
            groups.append(cls(codebook=cb, group_size=end - start, layer_range=(start, end - 1)))
        return groups

    def encode(self, vectors: Tensor) -> Tensor:
        return self.codebook.encode(vectors)

    def decode(self, codes: Tensor) -> Tensor:
        return self.codebook.decode(codes)


@dataclass
class ResidualPQCodebook:
    """Two-stage residual product quantization.

    Stage 1: encode with ``primary`` codebook.
    Stage 2: compute residual (original - stage1 reconstruction),
             encode residual with ``residual`` codebook.

    Reconstruction = primary.decode(codes1) + residual.decode(codes2).
    Typically halves reconstruction error vs single-stage PQ.

    Attributes:
        primary: first-stage PQ codebook.
        residual: second-stage PQ codebook trained on residuals.
    """

    primary: PQCodebook
    residual: PQCodebook

    @property
    def sub_dim(self) -> int:
        return self.primary.sub_dim

    @property
    def head_dim(self) -> int:
        return self.primary.head_dim

    @classmethod
    def train(
        cls,
        vectors: Tensor,
        sub_dim: int = DEFAULT_SUB_DIM,
        num_centroids: int = DEFAULT_NUM_CENTROIDS,
    ) -> ResidualPQCodebook:
        """Train a two-stage residual PQ codebook.

        Args:
            vectors: [N, head_dim] calibration activations.
            sub_dim: sub-vector length.
            num_centroids: entries per sub-codebook per stage.

        Returns:
            Trained ``ResidualPQCodebook``.
        """
        # Stage 1: train primary codebook.
        primary = PQCodebook.train(vectors, sub_dim=sub_dim, num_centroids=num_centroids)

        # Compute residuals.
        codes1 = primary.encode(vectors)
        recon1 = primary.decode(codes1)
        residuals = vectors.float() - recon1.float()

        # Stage 2: train residual codebook on the residuals.
        residual_cb = PQCodebook.train(residuals, sub_dim=sub_dim, num_centroids=num_centroids)

        return cls(primary=primary, residual=residual_cb)

    def encode(self, vectors: Tensor) -> tuple[Tensor, Tensor]:
        """Encode to (primary_codes, residual_codes), each [..., n_sub] uint8."""
        codes1 = self.primary.encode(vectors)
        recon1 = self.primary.decode(codes1)
        residuals = vectors.float() - recon1.float()
        codes2 = self.residual.encode(residuals)
        return codes1, codes2

    def decode(self, codes1: Tensor, codes2: Tensor) -> Tensor:
        """Decode: primary reconstruction + residual reconstruction."""
        return self.primary.decode(codes1) + self.residual.decode(codes2)

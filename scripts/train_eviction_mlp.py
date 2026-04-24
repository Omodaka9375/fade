"""Train the learned eviction policy MLP.

Collects (features, H2O-oracle-label) pairs from a calibration set and
trains the EvictionMLP to predict which tokens the H2O oracle would keep.

Usage:
    python scripts/train_eviction_mlp.py
    python scripts/train_eviction_mlp.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 20
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from fade.learned_policy import EvictionMLP, _build_features

# --- configuration ---------------------------------------------------------- #
DEFAULT_MODEL: str = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_EPOCHS: int = 10
DEFAULT_LR: float = 1e-3
DEFAULT_SAVE_PATH: str = "fade/checkpoints/eviction_mlp.pt"
N_SINK: int = 4
RECENT_WINDOW: int = 64
INT4_BUDGET: int = 400
NUM_LAYERS: int = 24  # typical for 0.5B-3B models
SEQ_LEN: int = 512
N_TRAIN_SAMPLES: int = 2000


def _generate_synthetic_data(
    n_samples: int, seq_len: int, num_layers: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic training data mimicking H2O oracle labels.

    Features: [N, seq_len, 4] (position, attn_mass, layer_depth, age).
    Labels: [N, seq_len] binary (1 = keep, 0 = evict).

    The synthetic oracle assigns higher keep-probability to:
        - Sink positions (first N_SINK)
        - Recent window (last RECENT_WINDOW)
        - Tokens with high attention mass
    """
    all_features = []
    all_labels = []

    for _ in range(n_samples):
        layer_idx = torch.randint(0, num_layers, (1,)).item()
        step = seq_len + torch.randint(0, 256, (1,)).item()

        # Synthetic attention scores: biased toward sinks and recent.
        scores = torch.rand(seq_len)
        scores[:N_SINK] += 2.0  # sinks get high attention
        scores[-RECENT_WINDOW:] += 1.5  # recent tokens get moderate boost

        features = _build_features(
            seq_len, scores, layer_idx, num_layers, step, torch.device("cpu")
        )

        # Oracle labels: keep sinks + recent + top-INT4_BUDGET by score.
        labels = torch.zeros(seq_len)
        labels[:N_SINK] = 1.0
        labels[-RECENT_WINDOW:] = 1.0
        middle = torch.zeros(seq_len, dtype=torch.bool)
        middle[N_SINK : seq_len - RECENT_WINDOW] = True
        middle_idx = middle.nonzero(as_tuple=False).squeeze(-1)
        if middle_idx.numel() > 0 and middle_idx.numel() > INT4_BUDGET:
            top_k = scores[middle_idx].topk(INT4_BUDGET).indices
            labels[middle_idx[top_k]] = 1.0
        elif middle_idx.numel() > 0:
            labels[middle_idx] = 1.0

        all_features.append(features)
        all_labels.append(labels)

    return torch.stack(all_features), torch.stack(all_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train eviction MLP")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--save", type=str, default=DEFAULT_SAVE_PATH)
    args = parser.parse_args()

    print("Generating synthetic training data...")
    features, labels = _generate_synthetic_data(N_TRAIN_SAMPLES, SEQ_LEN, NUM_LAYERS)
    # Flatten to [N*S, 4] and [N*S].
    X = features.view(-1, 4)
    y = labels.view(-1)

    mlp = EvictionMLP()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        mlp.train()
        pred = mlp(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy.
        with torch.no_grad():
            acc = ((pred > 0.5).float() == y).float().mean().item()
        print(f"  epoch {epoch + 1}/{args.epochs}  loss={loss.item():.4f}  acc={acc:.3f}")

    mlp.save(args.save)
    print(f"Saved to {args.save}")
    print(f"Parameters: {sum(p.numel() for p in mlp.parameters()):,}")


if __name__ == "__main__":
    main()

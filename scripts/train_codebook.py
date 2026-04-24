"""Train per-layer PQ codebooks from a calibration run.

Runs the model over calibration text, collects K/V activations per layer,
trains a PQCodebook per layer, and saves the result.

Usage:
    python scripts/train_codebook.py
    python scripts/train_codebook.py --model Qwen/Qwen2.5-0.5B-Instruct --out codebooks.pt
"""

from __future__ import annotations

import argparse

import torch

from fade.codebook import PQCodebook
from fade.patch import load_model

# --- configuration ---------------------------------------------------------- #
DEFAULT_MODEL: str = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUT: str = "codebooks.pt"
DEFAULT_SUB_DIM: int = 32
DEFAULT_NUM_CENTROIDS: int = 256
DEFAULT_CALIB_TOKENS: int = 2048
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32

CALIB_TEXT: str = (
    "The history of caching in computer systems spans several decades. "
    "Early mainframes used small buffers to avoid slow core memory accesses. "
    "Modern CPUs organize caches hierarchically, with L1, L2, and L3 levels. "
    "Language models reuse this idea when they keep key-value tensors across "
    "generation steps, avoiding redundant attention computation. "
    "Photosynthesis converts sunlight into chemical energy. Chloroplasts in "
    "plant cells capture photons, splitting water molecules to release oxygen. "
    "The Roman Empire at its height stretched from Britain to Mesopotamia. "
    "Quantum computers exploit superposition and entanglement to process "
    "information in fundamentally different ways than classical machines. "
) * 10


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PQ codebooks")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--sub-dim", type=int, default=DEFAULT_SUB_DIM)
    parser.add_argument("--centroids", type=int, default=DEFAULT_NUM_CENTROIDS)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model, device_map=DEVICE, dtype=DTYPE, attn_impl="eager")
    num_layers = model.config.num_hidden_layers

    # Tokenize calibration text.
    enc = tokenizer(
        CALIB_TEXT, return_tensors="pt", truncation=True, max_length=DEFAULT_CALIB_TOKENS
    )
    input_ids = enc.input_ids.to(DEVICE)
    print(f"Calibration tokens: {input_ids.shape[1]}")

    # Run forward to capture K/V from each layer.
    print("Collecting K/V activations...")
    from transformers import DynamicCache

    cache = DynamicCache()
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)

    # Train codebooks per layer.
    codebooks: dict[str, PQCodebook] = {}
    for layer_idx in range(num_layers):
        # DynamicCache stores key_cache[layer_idx] as [B, H, S, D].
        if hasattr(cache, "key_cache"):
            k = cache.key_cache[layer_idx]  # [B, H, S, D]
            v = cache.value_cache[layer_idx]
        else:
            layer = cache.layers[layer_idx]
            k = layer.keys if hasattr(layer, "keys") else layer.key_cache
            v = layer.values if hasattr(layer, "values") else layer.value_cache

        # Flatten to [N, D] for training.
        k_flat = k.float().cpu().reshape(-1, k.shape[-1])
        v_flat = v.float().cpu().reshape(-1, v.shape[-1])

        head_dim = k.shape[-1]
        sub_dim = min(args.sub_dim, head_dim)
        if head_dim % sub_dim != 0:
            # Find largest divisor <= sub_dim.
            for sd in range(sub_dim, 0, -1):
                if head_dim % sd == 0:
                    sub_dim = sd
                    break

        print(f"  Layer {layer_idx}/{num_layers}: K {k_flat.shape}, sub_dim={sub_dim}")
        k_cb = PQCodebook.train(k_flat, sub_dim=sub_dim, num_centroids=args.centroids)
        v_cb = PQCodebook.train(v_flat, sub_dim=sub_dim, num_centroids=args.centroids)
        codebooks[f"layer.{layer_idx}.k"] = k_cb
        codebooks[f"layer.{layer_idx}.v"] = v_cb

    # Save.
    save_dict = {}
    for name, cb in codebooks.items():
        save_dict[f"{name}.centroids"] = cb.centroids
        save_dict[f"{name}.sub_dim"] = cb.sub_dim
    save_dict["num_layers"] = num_layers
    torch.save(save_dict, args.out)
    print(f"Saved {len(codebooks)} codebooks to {args.out}")


if __name__ == "__main__":
    main()

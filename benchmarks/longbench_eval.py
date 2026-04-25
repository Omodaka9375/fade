"""LongBench evaluation for FADE.

Runs a subset of LongBench tasks (the standard 2026 benchmark for KV
compression papers) comparing baseline FP16 vs FADE presets.

Tasks evaluated:
  - Single-doc QA (qasper, multifieldqa_en)
  - Multi-doc QA (hotpotqa, 2wikimqa)
  - Summarization (gov_report, multi_news)

Reports per-task F1/ROUGE scores and an aggregate, matching the format
used by kvpress, TurboQuant, and KVTC papers.

Requires ``pip install fade-kv[eval]`` (pulls ``datasets``).

Usage:
    python benchmarks/longbench_eval.py
    python benchmarks/longbench_eval.py --model Qwen/Qwen2.5-7B-Instruct --presets safe balanced
    python benchmarks/longbench_eval.py --tasks qasper hotpotqa --max-samples 20
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from fade import FadeConfig, create_tiered_cache
from fade.patch import load_model
from fade.policy import reassign_tiers_by_position

# --- configuration (top of file for easy override) -------------------------- #
DEFAULT_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32

DEFAULT_TASKS: list[str] = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
]
DEFAULT_MAX_SAMPLES: int = 50
DEFAULT_MAX_NEW_TOKENS: int = 128
REASSIGN_EVERY: int = 64


# --- scoring ---------------------------------------------------------------- #
def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles for F1 computation."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def _f1_score(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference."""
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _rouge_l(prediction: str, reference: str) -> float:
    """Simple ROUGE-L (longest common subsequence) F1."""
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not ref_tokens or not pred_tokens:
        return 0.0

    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall)


def score_sample(prediction: str, references: list[str], task: str) -> float:
    """Score a single prediction against references."""
    if "report" in task or "news" in task:
        return max(_rouge_l(prediction, ref) for ref in references)
    return max(_f1_score(prediction, ref) for ref in references)


# --- data loading ----------------------------------------------------------- #
def load_longbench_task(task: str, max_samples: int) -> list[dict]:
    """Load a LongBench task from HuggingFace.

    Uses the auto-converted Parquet branch when available, falls back
    to downloading the JSONL directly from the dataset repo.
    """
    import json

    # Try Parquet branch first (works with all datasets versions).
    try:
        from datasets import load_dataset

        ds = load_dataset("THUDM/LongBench", task, split="test", revision="refs/convert/parquet")
        rows = list(ds)
    except Exception:
        # Fall back to downloading JSONL directly.
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "LongBench evaluation requires `huggingface_hub`. "
                "Install with: pip install fade-kv[eval]"
            ) from e

        local = hf_hub_download(
            repo_id="THUDM/LongBench",
            filename=f"data/{task}.jsonl",
            repo_type="dataset",
        )
        with open(local, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

    samples = []
    for i, row in enumerate(rows):
        if i >= max_samples:
            break
        answers = row.get("answers", [row.get("answer", "")])
        if isinstance(answers, str):
            answers = [answers]
        samples.append(
            {
                "context": row["context"],
                "input": row["input"],
                "answers": answers,
                "length": row.get("length", len(row["context"])),
            }
        )
    return samples


# --- generation ------------------------------------------------------------- #
@torch.no_grad()
def generate_with_fade(
    model,
    tokenizer,
    prompt: str,
    preset: str | None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """Generate a response using FADE cache (or baseline if preset is None)."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(DEVICE)

    if preset is None:
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:
        preset_fn = getattr(FadeConfig, preset, FadeConfig.safe)
        config = preset_fn()
        if config.eviction_policy == "h2o":
            config = config.with_overrides(eviction_policy="position")

        cache = create_tiered_cache(model, dtype=DTYPE, config=config)
        num_layers = model.config.num_hidden_layers

        # Prefill.
        out_obj = model(enc.input_ids, past_key_values=cache, use_cache=True)
        reassign_tiers_by_position(cache, num_layers)

        # Decode.
        next_tok = out_obj.logits[:, -1:, :].argmax(dim=-1)
        generated = [next_tok]
        for step in range(max_new_tokens - 1):
            out_obj = model(next_tok, past_key_values=cache, use_cache=True)
            next_tok = out_obj.logits[:, -1:, :].argmax(dim=-1)
            generated.append(next_tok)
            if (step + 1) % REASSIGN_EVERY == 0:
                reassign_tiers_by_position(cache, num_layers)
            if tokenizer.eos_token_id is not None and next_tok.item() == tokenizer.eos_token_id:
                break

        gen_ids = torch.cat(generated, dim=-1)
        return tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    return tokenizer.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True)


# --- main ------------------------------------------------------------------- #
def evaluate_task(
    model,
    tokenizer,
    task: str,
    preset: str | None,
    max_samples: int,
) -> dict:
    """Evaluate one task with one preset."""
    samples = load_longbench_task(task, max_samples)
    scores = []

    for i, sample in enumerate(samples):
        prompt = f"Context:\n{sample['context']}\n\nQuestion: {sample['input']}\nAnswer:"
        prediction = generate_with_fade(model, tokenizer, prompt, preset)
        s = score_sample(prediction, sample["answers"], task)
        scores.append(s)
        if (i + 1) % 10 == 0:
            print(f"    [{i + 1}/{len(samples)}] running avg: {sum(scores) / len(scores):.3f}")

    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "task": task,
        "preset": preset or "baseline",
        "n_samples": len(samples),
        "avg_score": round(avg * 100, 1),
        "scores": [round(s, 4) for s in scores],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LongBench evaluation for FADE")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--presets", type=str, nargs="+", default=[None, "safe", "balanced"])
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--out", type=str, default="benchmarks/longbench_results.json")
    args = parser.parse_args()

    # Normalize presets (allow "baseline" as alias for None).
    presets = [None if p in ("baseline", "none", "None") else p for p in args.presets]

    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model, device_map=DEVICE, dtype=DTYPE, attn_impl="sdpa")

    all_results = []

    for preset in presets:
        label = preset or "baseline"
        print(f"\n{'=' * 60}")
        print(f"  Preset: {label}")
        print(f"{'=' * 60}")

        task_scores = []
        for task in args.tasks:
            print(f"\n  Task: {task}")
            result = evaluate_task(model, tokenizer, task, preset, args.max_samples)
            task_scores.append(result)
            print(f"  → {result['avg_score']:.1f}")

        aggregate = (
            sum(r["avg_score"] for r in task_scores) / len(task_scores) if task_scores else 0
        )
        all_results.append(
            {
                "preset": label,
                "aggregate": round(aggregate, 1),
                "tasks": task_scores,
            }
        )
        print(f"\n  Aggregate: {aggregate:.1f}")

    # Summary table.
    print(f"\n{'=' * 60}")
    print("  LongBench Summary")
    print(f"{'=' * 60}")
    header = f"{'Preset':>12}"
    for task in args.tasks:
        header += f" {task[:12]:>12}"
    header += f" {'Aggregate':>12}"
    print(header)

    for res in all_results:
        row = f"{res['preset']:>12}"
        task_map = {t["task"]: t for t in res["tasks"]}
        for task in args.tasks:
            score = task_map.get(task, {}).get("avg_score", 0)
            row += f" {score:>11.1f}"
        row += f" {res['aggregate']:>11.1f}"
        print(row)

    # Save.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"model": args.model, "results": all_results},
            f,
            indent=2,
            default=str,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

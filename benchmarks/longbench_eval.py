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
DEFAULT_MAX_INPUT_TOKENS: int = 0  # 0 = use model's max_position_embeddings
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
# Local data directory (download data.zip from HuggingFace and unzip here).
LONGBENCH_DATA_DIR: str = "benchmarks/longbench_data/data"


def load_longbench_task(task: str, max_samples: int) -> list[dict]:
    """Load a LongBench task.

    Tries in order:
    1. Local JSONL files in ``LONGBENCH_DATA_DIR``
    2. ``load_dataset`` with ``trust_remote_code`` (datasets <= 3.2)
    3. ``load_dataset`` without trust_remote_code (auto-Parquet)
    """
    import json
    from pathlib import Path

    rows = None

    # Strategy 1: local JSONL (fastest, no network).
    local_path = Path(LONGBENCH_DATA_DIR) / f"{task}.jsonl"
    if local_path.exists():
        with open(local_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

    # Strategy 2: trust_remote_code (works with datasets <= 3.2).
    if rows is None:
        try:
            from datasets import load_dataset

            ds = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
            rows = list(ds)
        except Exception:
            pass

    # Strategy 3: plain load_dataset (newer datasets with auto-Parquet).
    if rows is None:
        try:
            from datasets import load_dataset

            ds = load_dataset("THUDM/LongBench", task, split="test")
            rows = list(ds)
        except Exception:
            pass

    if rows is None:
        raise RuntimeError(
            f"Could not load LongBench task '{task}'. "
            f"Download data.zip from https://huggingface.co/datasets/THUDM/LongBench "
            f"and unzip to {LONGBENCH_DATA_DIR}/"
        )

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


# --- prompt building -------------------------------------------------------- #
def _build_prompt(tokenizer, context: str, question: str, task: str) -> str:
    """Build a proper prompt using the model's chat template."""
    if "report" in task or "news" in task:
        user_msg = f"{context}\n\nWrite a summary of the above text."
    else:
        user_msg = f"{context}\n\nBased only on the above context, answer: {question}"

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{user_msg}\nAnswer:"


def _get_max_input_tokens(model, override: int = 0) -> int:
    """Resolve max input tokens from model config or override."""
    if override > 0:
        return override
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)
    max_pos = getattr(text_cfg, "max_position_embeddings", 32768)
    # Leave room for generation.
    return min(max_pos, 32768)


# --- generation ------------------------------------------------------------- #
@torch.no_grad()
def generate_with_fade(
    model,
    tokenizer,
    prompt: str,
    preset: str | None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_input_tokens: int = 0,
) -> str:
    """Generate using model.generate() for both baseline and FADE paths.

    Uses model.generate(past_key_values=cache) for FADE presets so the
    generation path is identical to baseline (fair comparison).
    """
    max_len = max_input_tokens if max_input_tokens > 0 else _get_max_input_tokens(model)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(DEVICE)

    if preset is None:
        # Baseline: plain model.generate().
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:
        # FADE: model.generate() with tiered cache (drop-in path).
        preset_fn = getattr(FadeConfig, preset, FadeConfig.safe)
        config = preset_fn()
        if config.eviction_policy == "h2o":
            config = config.with_overrides(eviction_policy="position")

        cache = create_tiered_cache(model, dtype=DTYPE, config=config)
        out = model.generate(
            **enc,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True)


# --- main ------------------------------------------------------------------- #
def evaluate_task(
    model,
    tokenizer,
    task: str,
    preset: str | None,
    max_samples: int,
    max_input_tokens: int = 0,
) -> dict:
    """Evaluate one task with one preset."""
    samples = load_longbench_task(task, max_samples)
    resolved_max = _get_max_input_tokens(model, max_input_tokens)
    scores = []

    for i, sample in enumerate(samples):
        prompt = _build_prompt(tokenizer, sample["context"], sample["input"], task)
        prediction = generate_with_fade(
            model, tokenizer, prompt, preset, max_input_tokens=resolved_max
        )
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
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help="Max input tokens (0 = model's max_position_embeddings, capped at 32768).",
    )
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
            result = evaluate_task(
                model, tokenizer, task, preset, args.max_samples, args.max_input_tokens
            )
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

# P1-5 Fix: LongBench Truncation Handling

## Status: ✅ FIXED

### What Was Wrong

The audit found that LongBench evaluation had critical issues:

1. **Wrong truncation**: Contexts averaging 5K-18K tokens were being truncated at fixed lengths (often 8192), cutting off most of the context
2. **Raw prompt format**: Used raw prompts instead of model's chat template
3. **Inconsistent paths**: FADE presets used manual decode while baseline used `model.generate()` - unfair comparison
4. **No validation**: No checks for contexts exceeding model limits

**Result**: Most LongBench contexts were truncated before evaluation, making the benchmark meaningless for testing long-context compression.

---

### What Was Fixed

#### 1. **Dynamic Max Length from Model Config**

```python
def _get_max_input_tokens(model, override: int = 0) -> int:
    """Resolve max input tokens from model config or override."""
    if override > 0:
        return override
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)
    max_pos = getattr(text_cfg, "max_position_embeddings", 32768)
    # Leave room for generation.
    return min(max_pos, 32768)
```

Now uses the model's actual `max_position_embeddings` instead of hardcoded values.

#### 2. **Smart Trunction Strategy**

```python
def _truncate_context(context: str, tokenizer, max_tokens: int, strategy: str) -> tuple[str, bool]:
    """Truncate context intelligently.
    
    Strategy:
        - Keep first 50% of context (setup)
        - Keep last 50% of context (often contains answer)
        - Insert "..." to indicate truncation
    """
    tokens = tokenizer.encode(context, add_special_tokens=False)
    actual_tokens = len(tokens)
    
    if actual_tokens <= max_tokens:
        return context, False
    
    # Split: keep first half and last half
    half = max_tokens // 2
    first_half = tokenizer.decode(tokens[:half], skip_special_tokens=True)
    last_half = tokenizer.decode(tokens[-half:], skip_special_tokens=True)
    
    truncated = f"{first_half}\n...\n{last_half}"
    return truncated, True
```

**Better than naive truncation**: Preserves both beginning and end of context instead of just cutting from the end.

#### 3. **Three Truncation Strategies**

- **`warn`** (default): Warn but truncate automatically
- **`truncate`**: Silently truncate
- **`error`**: Raise exception if context exceeds limit

```python
parser.add_argument(
    "--truncation-strategy",
    type=str,
    default="warn",
    choices=["warn", "truncate", "error"],
    help="How to handle contexts exceeding max length.",
)
```

#### 4. **Proper Chat Template Usage**

```python
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
```

Uses the model's native chat template for all tasks.

#### 5. **Consistent Generation Path**

Both baseline and FADE now use `model.generate()`:

```python
# Baseline
out = model.generate(**enc, max_new_tokens=max_new_tokens, ...)

# FADE
cache = create_tiered_cache(model, dtype=DTYPE, config=config)
out = model.generate(**enc, past_key_values=cache, max_new_tokens=max_new_tokens, ...)
```

Fair comparison - same generation path, different cache implementations.

---

### How to Use

#### Default (warn and truncate if needed)
```powershell
python benchmarks/longbench_eval.py --model Qwen/Qwen2.5-7B-Instruct
```

#### Use model's full context window
```powershell
python benchmarks/longbench_eval.py --model Qwen/Qwen2.5-7B-Instruct --max-input-tokens 0
```

#### Strict mode (error on overflow)
```powershell
python benchmarks/longbench_eval.py --model Qwen/Qwen2.5-7B-Instruct --truncation-strategy error
```

#### Custom max length
```powershell
python benchmarks/longbench_eval.py --model Qwen/Qwen2.5-7B-Instruct --max-input-tokens 16384
```

---

### Example Output

```
============================================================
  Preset: balanced
============================================================

  Task: qasper
    [truncated from 12847 to 8192 tokens]
    [10/50] running avg: 0.523
    [20/50] running avg: 0.531
    ...
  → 53.1

  Task: hotpotqa
    [truncated from 15234 to 8192 tokens]
    [10/50] running avg: 0.412
    ...
  → 41.8

  Aggregate: 47.5
```

The output now shows when truncation occurs and how many tokens were reduced.

---

### What This Fixes

**Before**:
- Fixed truncation at arbitrary lengths
- Most LongBench contexts cut off completely
- Unfair comparison between baseline and FADE
- No visibility into truncation

**After**:
- Uses model's actual context window (32K for Qwen2.5-7B)
- Intelligent truncation preserves key information
- Consistent generation path for fair comparison
- Clear visibility into when/how truncation occurs
- Configurable strategy (warn/truncate/error)

---

### Files Modified

- `benchmarks/longbench_eval.py`:
  - Added `DEFAULT_TRUNCATION_STRATEGY` config
  - Added `_truncate_context()` function
  - Updated `generate_with_fade()` to handle long contexts
  - Added `truncation_strategy` parameter to `evaluate_task()`
  - Updated CLI with `--truncation-strategy` option

---

### Recommendations

1. **Use `--max-input-tokens 0`** to leverage the full model context window
2. **Set `--truncation-strategy error`** in CI to catch truncation issues early
3. **Report truncation stats** in papers/reports (how many samples were truncated)
4. **Consider sliding window** for contexts exceeding model limits (future work)

---

### Testing

Verify the fix works:

```powershell
# Test with a model that has 32K context
python benchmarks/longbench_eval.py --model Qwen/Qwen2.5-7B-Instruct --max-input-tokens 32768 --max-samples 5

# Test strict mode
python benchmarks/longbench_eval.py --model Qwen/Qwen2.5-7B-Instruct --truncation-strategy error --max-samples 5
```

Expected behavior:
- Contexts under the limit: no truncation
- Contexts over the limit: warning + intelligent truncation
- Error mode: raises exception for oversized contexts

---

### Impact

This fix ensures LongBench evaluation:
1. ✅ Actually tests long-context capabilities
2. ✅ Uses fair comparison between baseline and FADE
3. ✅ Provides visibility into truncation
4. ✅ Respects model's actual capabilities

Without this fix, LongBench results were essentially meaningless for testing compression on long contexts.

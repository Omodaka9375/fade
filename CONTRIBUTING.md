# Contributing to FADE
Thanks for your interest. FADE is a tiered KV-cache compression library for
transformer inference; contributions that improve correctness, coverage, or
performance are welcome.
## Dev setup
```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Install torch FIRST with the CUDA build matching your GPU:
# https://pytorch.org/get-started/locally/
pip install -e ".[dev,eval]"
pre-commit install
```
## Running the checks
```pwsh
ruff check .
ruff format --check .
pyright fade
pytest
```
All four must pass before a PR can be merged. CI runs them on the full support
matrix (Python 3.10 / 3.12 × transformers 4.45 / 5.3).
## Testing guidelines
- **Unit tests** (`tests/test_quant.py`, `tests/test_cache.py`) must stay CPU-only
  and finish in under a second each.
- **Integration tests** (`tests/test_integration.py`) use tiny random-init models
  from `AutoConfig.for_model(...)`. **No HuggingFace downloads in tests.**
- **Quality tests** (`tests/test_eval/`, marked with `@pytest.mark.eval`) may
  download models and take minutes; they're gated behind `pytest -m eval` and
  only run on nightly CI.
- Add a regression test for every bug you fix. Link it from the PR description.
## Pull request checklist
- [ ] `pre-commit run --all-files` passes.
- [ ] `pytest` passes on your local machine.
- [ ] New public APIs have concise docstrings (Hemingway-test: short sentences).
- [ ] User-visible behavior changes update `README.md`.
- [ ] CHANGELOG entry under `[Unreleased]`.
## Code style
- Python 3.10+; use `from __future__ import annotations`.
- Prefer `type` over `interface`-style class hierarchies for dumb data — use
  `@dataclass` or `TypedDict`.
- All tunable knobs at the top of the file as module constants, typed.
- Keep JS-style docstrings out; follow NumPy/Google docstring conventions.
## Scope
FADE is inference-only. We explicitly do NOT accept:
- Training-time KV compression.
- Multi-GPU tensor-parallel sharding of the cache (delegated to vLLM / SGLang).
- Custom CUDA kernels outside of Triton (maintenance overhead too high).
## Reporting issues
Open a GitHub issue with: a minimal reproduction, the model ID, the
transformers / torch / CUDA versions, and the output of `run_tiered.py` with
`PROMPT_MODE="chat"` if the bug is quality-related.

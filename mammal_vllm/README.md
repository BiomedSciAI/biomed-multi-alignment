# mammal-vllm

A **vLLM plugin** that exposes the [IBM MAMMAL biomedical foundation model](https://huggingface.co/ibm-research/biomed.omics.bl.sm.ma-ted-458m) as an **encoder-only embedding model** inside vLLM's pooling runner.

MAMMAL is a 458M-parameter T5-style encoder-decoder model trained on over 2 billion biological samples across proteins, small molecules, and single-cell gene expression data. This plugin uses **only the encoder stack** and applies mean-pooled, L2-normalized hidden states as dense embedding vectors.

## Key Features

- 🧬 **Multi-modal biomedical embeddings**: Proteins, small molecules, and gene expression
- ⚡ **High-performance inference**: Leverages vLLM's optimized pooling runner
- 🔌 **Easy integration**: Auto-discovered plugin via Python entry points
- 🎯 **Encoder-only**: Uses only the T5 encoder stack for efficient embeddings
- 📊 **Normalized embeddings**: L2-normalized for direct similarity comparisons

---

## Installation

### Prerequisites

This plugin requires [uv](https://docs.astral.sh/uv/) for package management. If you don't have it installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
### From source

```bash
# Navigate to the mammal_vllm directory
cd /path/to/biomed-multi-alignment/mammal_vllm

# Install with all dependencies
pip install -e .
```

### Verification

After installation, verify the plugin is registered:

```python
from vllm_mammal_plugin import register_mammal_model
print("Plugin successfully installed!")
```

vLLM auto-discovers the plugin via the `entry_points` mechanism defined in `pyproject.toml`.

---

## Usage

### Offline Inference (Python API)

Use vLLM's Python API directly for embedding generation. See the complete example:

**📄 [`examples/offline_mammal_usage.py`](examples/offline_mammal_usage.py)**

Key points:
- Initialize LLM with `runner="pooling"` and `tokenizer_mode="mammal"`
- Pass plain text prompts directly to `model.embed()` — `MammalTokenizer` handles tokenization automatically
- Embeddings are L2-normalized

### Online Serving (OpenAI-Compatible API)

Start a vLLM server and use the OpenAI-compatible client. See the complete example:

**📄 [`examples/online_mammal_usage.py`](examples/online_mammal_usage.py)**

Server command:
```bash
vllm serve ibm-research/biomed.omics.bl.sm.ma-ted-458m \
    --runner pooling \
    --trust-remote-code \
    --tokenizer_mode mammal \
    --gpu_memory_utilization 0.4 \
    --enforce_eager \
    --no-enable-prefix-caching
```

Key points:
- Pass plain text prompts directly — the server tokenizes via `MammalTokenizer` when started with `--tokenizer_mode mammal`
- Use the OpenAI client with the `/v1/embeddings` endpoint
- Embeddings are L2-normalized

---

## MAMMAL Input Format

See [`examples/example_prompts.py`](examples/example_prompts.py) for pre-formatted example prompts.

---

## Testing

### Run all tests

```bash
pytest tests/ -v
```

### Run only CPU tests (no GPU, no model download)

```bash
pytest tests/test_registration.py tests/test_example_prompts.py -v
```

### Run GPU tests explicitly

```bash
pytest tests/test_embeddings.py tests/test_compare_embeddings.py -v
```

### Embedding comparison benchmark

`compare_embeddings.py` is a standalone benchmark script (not collected by
pytest) that prints timing and similarity metrics for all modalities:

```bash
# Offline vLLM vs direct MAMMAL
python tests/compare_embeddings.py

# Also compare against a running vLLM server
COMPARE_ONLINE=true python tests/compare_embeddings.py
```

## Project Structure

```
mammal_vllm/
├── vllm_mammal_plugin/
│   ├── __init__.py              # Plugin registration, tokenizer + renderer registration
│   ├── mammal.py                # Model implementation
│   └── tokenization.py         # MammalTokenizer (TokenizerLike wrapper)
├── examples/
│   ├── __init__.py              # Package marker
│   ├── example_prompts.py       # Pre-formatted example prompts
│   ├── offline_mammal_usage.py  # Offline inference example
│   └── online_mammal_usage.py   # Online serving example
├── tests/
│   ├── conftest.py              # pytest config: sys.path, requires_gpu marker
│   ├── compare_embeddings.py    # Embedding standalone benchmark script
│   ├── test_registration.py     # Plugin importability (CPU-only)
│   ├── test_example_prompts.py  # Prompt constant structure (CPU-only)
│   ├── test_tokenization.py     # MammalTokenizer (CPU-only)
│   ├── test_embeddings.py       # vLLM offline embeddings (GPU)
│   └── test_compare_embeddings.py  # vLLM vs MAMMAL comparison (GPU)
├── __init__.py                  # Root package marker
├── pyproject.toml               # Project configuration
├── setup.py                     # Setup script
└── README.md                    # This file
```

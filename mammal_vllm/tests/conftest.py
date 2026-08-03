"""
conftest.py — pytest configuration for mammal_vllm tests.

Adds the package root to sys.path so that ``examples`` and
``vllm_mammal_plugin`` are importable without a prior ``pip install -e .``.
Registers the ``requires_gpu`` mark, which skips any test when CUDA is not
available.
"""

import sys
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Make package root importable
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).parent.parent  # …/mammal_vllm/
_TESTS_DIR = Path(__file__).parent        # …/mammal_vllm/tests/
for _p in (_PKG_ROOT, _TESTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# Custom marks
# ---------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_gpu: mark test as requiring a CUDA-capable GPU "
        "(automatically skipped when torch.cuda.is_available() is False).",
    )


# ---------------------------------------------------------------------------
# Auto-skip requires_gpu tests when no GPU is present
# ---------------------------------------------------------------------------
def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("requires_gpu") and not torch.cuda.is_available():
        pytest.skip("requires a CUDA-capable GPU")

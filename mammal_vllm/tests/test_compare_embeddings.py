"""
tests/test_compare_embeddings.py
---------------------------------
Compare embeddings produced by the vLLM plugin against those produced by
the direct MAMMAL model.

.. note::
    All tests in this module require a CUDA-capable GPU **and** both the
    ``vllm-mammal-plugin`` and ``mammal`` (biomed-multi-alignment) packages
    installed.  They are automatically skipped when ``torch.cuda.is_available()``
    is ``False`` (see conftest.py).

Optional online comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~
Set the environment variable ``COMPARE_ONLINE=true`` to also compare against
a running vLLM server.  Start the server first::

    vllm serve ibm-research/biomed.omics.bl.sm.ma-ted-458m \
        --runner pooling \
        --trust-remote-code \
        --tokenizer_mode mammal \
        --gpu_memory_utilization 0.4 \
        --enforce_eager \
        --no-enable-prefix-caching
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from compare_embeddings import (
    MODEL_NAME,
    cosine_similarity,
    get_mammal_embeddings,
    get_online_vllm_embeddings,
    get_vllm_embeddings,
)
from examples.example_prompts import (
    GENE_BRCA1,
    GENE_MALAT1,
    PROTEIN_CALMODULIN,
    PROTEIN_FLUORESCENT,
    SMILES_ASPIRIN,
    SMILES_CAFFEINE,
    SMILES_ETHER,
)

ALL_PROMPTS: list[str] = [
    PROTEIN_CALMODULIN,
    SMILES_ASPIRIN,
    SMILES_CAFFEINE,
    PROTEIN_FLUORESCENT,
    SMILES_ETHER,
    GENE_MALAT1,
    GENE_BRCA1,
]

ALL_PROMPT_NAMES: list[str] = [
    "Calmodulin (protein)",
    "Aspirin (SMILES)",
    "Caffeine (SMILES)",
    "Fluorescent (protein)",
    "Ether (SMILES)",
    "Malat1 (gene)",
    "BRCA1 (gene)",
]


# ---------------------------------------------------------------------------
# Session-scoped fixtures — models loaded once per pytest run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def vllm_embeddings() -> list[np.ndarray]:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-capable GPU")
    embeddings, _, _ = get_vllm_embeddings(ALL_PROMPTS)
    return embeddings


@pytest.fixture(scope="session")
def mammal_embeddings() -> list[np.ndarray]:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-capable GPU")
    embeddings, _, _ = get_mammal_embeddings(MODEL_NAME, ALL_PROMPTS)
    return embeddings


@pytest.fixture(scope="session")
def online_vllm_embeddings() -> list[np.ndarray] | None:
    """Returns online embeddings only when COMPARE_ONLINE env-var is set; else None."""
    if os.environ.get("COMPARE_ONLINE", "").lower() not in ("true", "1", "yes"):
        return None
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-capable GPU")
    try:
        embeddings, _ = get_online_vllm_embeddings(ALL_PROMPTS)
        return embeddings
    except Exception:
        pytest.skip("COMPARE_ONLINE requested but online vLLM server not reachable")
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_gpu
class TestEmbeddingComparison:
    """
    Compare embeddings from the vLLM plugin against the direct MAMMAL model.

    .. note::
        Requires a CUDA-capable GPU and both ``vllm`` and
        ``biomed-multi-alignment`` installed.
        Automatically skipped without a GPU (see conftest.py).
    """

    @pytest.mark.parametrize(("idx", "name"), list(enumerate(ALL_PROMPT_NAMES)))
    def test_vllm_vs_mammal_cosine_similarity(
        self,
        vllm_embeddings: list[np.ndarray],
        mammal_embeddings: list[np.ndarray],
        idx: int,
        name: str,
    ) -> None:
        """Cosine similarity between vLLM and MAMMAL embeddings must exceed 0.95."""
        sim = cosine_similarity(vllm_embeddings[idx], mammal_embeddings[idx])
        assert (
            sim > 0.95
        ), f"{name}: vLLM vs MAMMAL cosine similarity too low ({sim:.6f})"

    @pytest.mark.parametrize(("idx", "name"), list(enumerate(ALL_PROMPT_NAMES)))
    def test_vllm_vs_mammal_embedding_shapes_match(
        self,
        vllm_embeddings: list[np.ndarray],
        mammal_embeddings: list[np.ndarray],
        idx: int,
        name: str,
    ) -> None:
        """vLLM and MAMMAL must produce embeddings with identical shapes."""
        assert vllm_embeddings[idx].shape == mammal_embeddings[idx].shape, (
            f"{name}: shape mismatch — "
            f"vLLM {vllm_embeddings[idx].shape} vs MAMMAL {mammal_embeddings[idx].shape}"
        )

    @pytest.mark.parametrize(("idx", "name"), list(enumerate(ALL_PROMPT_NAMES)))
    def test_online_vs_mammal_cosine_similarity(
        self,
        online_vllm_embeddings: list[np.ndarray] | None,
        mammal_embeddings: list[np.ndarray],
        idx: int,
        name: str,
    ) -> None:
        """Online vLLM vs MAMMAL cosine similarity must exceed 0.95 (skipped if server absent)."""
        if online_vllm_embeddings is None:
            pytest.skip("COMPARE_ONLINE not set — online server comparison skipped")
        sim = cosine_similarity(online_vllm_embeddings[idx], mammal_embeddings[idx])
        assert (
            sim > 0.95
        ), f"{name}: online vLLM vs MAMMAL cosine similarity too low ({sim:.6f})"

    def test_vllm_embeddings_are_finite(
        self, vllm_embeddings: list[np.ndarray]
    ) -> None:
        """All vLLM embeddings must be finite (no NaN or Inf values)."""
        for emb, name in zip(vllm_embeddings, ALL_PROMPT_NAMES):
            assert not np.isnan(emb).any(), f"{name}: vLLM embedding contains NaN"
            assert not np.isinf(emb).any(), f"{name}: vLLM embedding contains Inf"

    def test_mammal_embeddings_are_finite(
        self, mammal_embeddings: list[np.ndarray]
    ) -> None:
        """All direct MAMMAL embeddings must be finite (no NaN or Inf values)."""
        for emb, name in zip(mammal_embeddings, ALL_PROMPT_NAMES):
            assert not np.isnan(emb).any(), f"{name}: MAMMAL embedding contains NaN"
            assert not np.isinf(emb).any(), f"{name}: MAMMAL embedding contains Inf"


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

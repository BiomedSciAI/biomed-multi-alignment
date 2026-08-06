"""
tests/test_embeddings.py
------------------------
Test vLLM offline embedding generation.

.. note::
    All tests in this module require a CUDA-capable GPU and the ``vllm``
    package installed.  They are automatically skipped when
    ``torch.cuda.is_available()`` is ``False`` (see conftest.py).
"""

import numpy as np
import pytest
from examples.example_prompts import (
    GENE_BRCA1,
    PROTEIN_CALMODULIN,
    SMILES_ASPIRIN,
)


@pytest.mark.requires_gpu
class TestVLLMEmbeddings:
    """
    Test vLLM offline embedding generation.

    .. note::
        Requires a CUDA-capable GPU and the ``vllm`` package installed.
        Automatically skipped when ``torch.cuda.is_available()`` is ``False``.
    """

    MODEL_NAME = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"
    EMBEDDING_DIM = 768

    @pytest.fixture(scope="class")
    @classmethod
    def llm(cls):
        """Shared LLM instance — loaded once per test class to save GPU memory."""
        from vllm import LLM

        return LLM(
            model=cls.MODEL_NAME,
            runner="pooling",
            trust_remote_code=True,
            tokenizer_mode="mammal",
            gpu_memory_utilization=0.4,
            enforce_eager=True,
            enable_prefix_caching=False,
        )

    @pytest.mark.parametrize(
        ("prompt", "label"),
        [
            (PROTEIN_CALMODULIN, "Calmodulin"),
            (SMILES_ASPIRIN, "Aspirin"),
            (GENE_BRCA1, "BRCA1"),
        ],
    )
    def test_embedding_shape_and_validity(self, llm, prompt: str, label: str) -> None:
        """Each prompt must produce a finite, L2-normalised embedding of the expected shape."""
        outputs = llm.embed([prompt])
        embedding = np.array(outputs[0].outputs.embedding)

        assert embedding.shape == (
            self.EMBEDDING_DIM,
        ), f"{label}: unexpected shape {embedding.shape}"
        assert not np.isnan(embedding).any(), f"{label}: embedding contains NaN"
        assert not np.isinf(embedding).any(), f"{label}: embedding contains Inf"

        norm = np.linalg.norm(embedding)
        assert (
            0.99 < norm < 1.01
        ), f"{label}: embedding not L2-normalised (norm={norm:.6f})"

    def test_different_modalities_produce_distinct_embeddings(self, llm) -> None:
        """Embeddings from different modalities must not be nearly identical."""
        prompts = [PROTEIN_CALMODULIN, SMILES_ASPIRIN, GENE_BRCA1]
        labels = ["Calmodulin", "Aspirin", "BRCA1"]

        outputs = llm.embed(prompts)
        embeddings = [np.array(o.outputs.embedding) for o in outputs]

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                cosine_sim = float(np.dot(embeddings[i], embeddings[j]))
                assert cosine_sim < 0.9, (
                    f"Embeddings for {labels[i]} and {labels[j]} are too similar: "
                    f"cosine_sim={cosine_sim:.6f}"
                )


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for the mammal plugin.
These tests do NOT require a running vLLM server.
"""

import numpy as np
import pytest
import torch
from vllm import LLM

from examples.example_prompts import (  # type: ignore[no-redef]
    GENE_BRCA1,
    GENE_MALAT1,
    PROTEIN_CALMODULIN,
    PROTEIN_FLUORESCENT,
    SMILES_ASPIRIN,
    SMILES_CAFFEINE,
)
from vllm_mammal_plugin.tokenization import MammalTokenizer


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------
class TestPluginRegistration:
    """Test that the plugin is properly registered with vLLM."""

    def test_register_function_exists(self):
        """Verify the registration function is callable."""
        from vllm_mammal_plugin import register_mammal_model

        assert callable(register_mammal_model)

    def test_model_class_importable(self):
        """Verify the T5ForConditionalGeneration class can be imported."""
        from vllm_mammal_plugin.mammal import T5ForConditionalGeneration

        assert T5ForConditionalGeneration is not None
        assert hasattr(T5ForConditionalGeneration, "load_weights")
        assert hasattr(T5ForConditionalGeneration, "forward")

    def test_config_class_importable(self):
        """Verify the MammalConfig class can be imported."""
        from vllm_mammal_plugin.mammal import MammalConfig

        assert MammalConfig is not None
        assert MammalConfig.model_type == "t5"


# ---------------------------------------------------------------------------
# Tokenization utilities
# ---------------------------------------------------------------------------
class TestTokenization:
    """Test MammalTokenizer without requiring GPU.

    A single tokenizer instance is shared across all tests via a class-level
    fixture to avoid redundant downloads/loads.
    """

    MODEL_NAME = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return MammalTokenizer.from_pretrained(self.MODEL_NAME)

    def test_tokenizer_importable(self):
        """Verify MammalTokenizer can be imported and constructed."""
        assert callable(MammalTokenizer.from_pretrained)

    def test_encode_all_modalities(self, tokenizer):
        """Encoding protein, SMILES, and gene prompts all produce non-empty int lists,
        and different inputs produce different token sequences."""
        tokens = {
            "protein": tokenizer.encode(PROTEIN_CALMODULIN),
            "smiles": tokenizer.encode(SMILES_ASPIRIN),
            "gene": tokenizer.encode(GENE_BRCA1),
        }
        for modality, ids in tokens.items():
            assert isinstance(ids, list) and len(ids) > 0, modality
            assert all(isinstance(t, int) for t in ids), modality

        assert tokens["protein"] != tokens["smiles"]
        assert tokens["smiles"] != tokens["gene"]

    def test_protocol_properties(self, tokenizer):
        """MammalTokenizer satisfies the TokenizerLike protocol."""
        assert isinstance(tokenizer.vocab_size, int) and tokenizer.vocab_size > 0
        assert isinstance(tokenizer.max_token_id, int) and tokenizer.max_token_id > 0
        assert isinstance(tokenizer.pad_token_id, int)
        assert isinstance(tokenizer.eos_token_id, int)
        assert tokenizer.is_fast is True
        assert tokenizer.truncation_side == "right"
        assert isinstance(tokenizer.all_special_tokens, list)
        assert len(tokenizer.all_special_tokens) > 0
        assert isinstance(tokenizer.get_vocab(), dict)
        assert len(tokenizer.get_vocab()) > 0

    def test_call_returns_batch_encoding(self, tokenizer):
        """__call__ returns a BatchEncoding with input_ids (used by vLLM's renderer)."""
        encoding = tokenizer(PROTEIN_CALMODULIN)
        assert "input_ids" in encoding
        assert isinstance(encoding["input_ids"], list)
        assert len(encoding["input_ids"]) > 0


# ---------------------------------------------------------------------------
# MAMMAL prompt formatting helpers (used in examples / application code)
# ---------------------------------------------------------------------------
def format_mammal_prompt(sequence: str, modality: str) -> str:
    """Reference implementation of MAMMAL prompt formatting.

    Note: The working example uses pre-formatted strings with <EOS> tokens.
    This helper is for testing the formatting logic only.
    """
    modality = modality.lower()
    if modality in ("protein", "aa", "amino_acid"):
        return (
            f"<@TOKENIZER-TYPE=AA>"
            f"<MOLECULAR_ENTITY><MOLECULAR_ENTITY_OF_TYPE_PROTEIN>"
            f"{sequence.strip()}"
            f"<EOS>"
        )
    if modality in ("smiles", "small_molecule", "drug"):
        return (
            f"<@TOKENIZER-TYPE=SMILES>"
            f"<MOLECULAR_ENTITY><MOLECULAR_ENTITY_OF_TYPE_SMALL_MOL>"
            f"{sequence.strip()}"
            f"<EOS>"
        )
    if modality in ("gene", "gene_expression"):
        return (
            f"<@TOKENIZER-TYPE=GENE>"
            f"<MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>"
            f"{sequence.strip()}"
            f"<EOS>"
        )
    return sequence


class TestPromptFormatting:
    """Test MAMMAL prompt formatting helpers."""

    def test_protein_prefix(self):
        """Test protein sequence formatting."""
        result = format_mammal_prompt("ACDEFG", "protein")
        assert result.startswith("<@TOKENIZER-TYPE=AA>")
        assert "<MOLECULAR_ENTITY_OF_TYPE_PROTEIN>" in result
        assert "ACDEFG" in result
        assert result.endswith("<EOS>")

    def test_smiles_prefix(self):
        """Test SMILES formatting."""
        result = format_mammal_prompt("CC(=O)O", "smiles")
        assert result.startswith("<@TOKENIZER-TYPE=SMILES>")
        assert "<MOLECULAR_ENTITY_OF_TYPE_SMALL_MOL>" in result
        assert result.endswith("<EOS>")

    def test_gene_prefix(self):
        """Test gene expression formatting."""
        result = format_mammal_prompt("BRCA1", "gene")
        assert result.startswith("<@TOKENIZER-TYPE=GENE>")
        assert "<MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>" in result
        assert "BRCA1" in result
        assert result.endswith("<EOS>")

    def test_text_passthrough(self):
        """Test that unknown modalities pass through unchanged."""
        seq = "raw gene expression data"
        assert format_mammal_prompt(seq, "text") == seq

    def test_protein_aliases(self):
        """Test protein modality aliases."""
        for alias in ("aa", "amino_acid"):
            result = format_mammal_prompt("AAA", alias)
            assert "<MOLECULAR_ENTITY_OF_TYPE_PROTEIN>" in result
            assert result.endswith("<EOS>")

    def test_smiles_aliases(self):
        """Test SMILES modality aliases."""
        for alias in ("small_molecule", "drug"):
            result = format_mammal_prompt("CC", alias)
            assert "<MOLECULAR_ENTITY_OF_TYPE_SMALL_MOL>" in result
            assert result.endswith("<EOS>")

    def test_whitespace_stripped(self):
        """Test that whitespace is properly stripped."""
        result = format_mammal_prompt("  ACDEFG  ", "protein")
        assert "ACDEFG<EOS>" in result

    def test_example_prompts_valid(self):
        """Verify example prompts have correct structure."""
        # Protein examples
        assert "<@TOKENIZER-TYPE=AA>" in PROTEIN_CALMODULIN
        assert "<MOLECULAR_ENTITY_OF_TYPE_PROTEIN>" in PROTEIN_CALMODULIN
        assert PROTEIN_CALMODULIN.endswith("<EOS>")

        assert "<@TOKENIZER-TYPE=AA>" in PROTEIN_FLUORESCENT
        assert PROTEIN_FLUORESCENT.endswith("<EOS>")

        # SMILES examples
        assert "<@TOKENIZER-TYPE=SMILES>" in SMILES_ASPIRIN
        assert "<MOLECULAR_ENTITY_OF_TYPE_SMALL_MOL>" in SMILES_ASPIRIN
        assert SMILES_ASPIRIN.endswith("<EOS>")

        assert "<@TOKENIZER-TYPE=SMILES>" in SMILES_CAFFEINE
        assert SMILES_CAFFEINE.endswith("<EOS>")

        # Gene examples
        assert "<@TOKENIZER-TYPE=GENE>" in GENE_BRCA1
        assert "<MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>" in GENE_BRCA1
        assert GENE_BRCA1.endswith("<EOS>")

        assert "<@TOKENIZER-TYPE=GENE>" in GENE_MALAT1
        assert GENE_MALAT1.endswith("<EOS>")


# ---------------------------------------------------------------------------
# vLLM Embeddings (requires GPU and vLLM)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
class TestVLLMEmbeddings:
    """Test vLLM embedding generation (requires GPU and vLLM installed)."""

    def test_get_vllm_embeddings(self) -> None:
        """Test that vLLM can generate embeddings for different modalities."""
        model_name = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"

        # Initialize vLLM model — tokenizer_mode="mammal" enables plain-text input
        llm = LLM(
            model=model_name,
            runner="pooling",
            trust_remote_code=True,
            tokenizer_mode="mammal",
            gpu_memory_utilization=0.4,
            enforce_eager=True,
            enable_prefix_caching=False,
        )

        # Test different modalities using plain text prompts
        test_cases = [
            ("protein", PROTEIN_CALMODULIN, "Calmodulin"),
            ("smiles", SMILES_ASPIRIN, "Aspirin"),
            ("gene", GENE_BRCA1, "BRCA1"),
        ]

        embeddings = []
        for modality, prompt, name in test_cases:
            outputs = llm.embed([prompt])
            embedding = np.array(outputs[0].outputs.embedding)
            embeddings.append(embedding)

            # Verify embedding properties
            assert embedding.shape == (
                768,
            ), f"{name} embedding has wrong shape: {embedding.shape}"
            assert not np.isnan(embedding).any(), f"{name} embedding contains NaN"
            assert not np.isinf(embedding).any(), f"{name} embedding contains Inf"

            # Check if normalized (L2 norm should be close to 1)
            norm = np.linalg.norm(embedding)
            assert 0.99 < norm < 1.01, f"{name} embedding not normalized: norm={norm}"

            print(f"✓ {name} ({modality}): shape={embedding.shape}, norm={norm:.6f}")

        # Verify different modalities produce different embeddings
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                cosine_sim = np.dot(embeddings[i], embeddings[j])
                assert (
                    cosine_sim < 0.9
                ), f"Embeddings {i} and {j} are too similar: {cosine_sim:.6f}"
                print(
                    f"✓ Embeddings {i} and {j} are distinct: similarity={cosine_sim:.6f}"
                )


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

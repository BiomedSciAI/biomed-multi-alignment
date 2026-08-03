"""
tests/test_tokenization.py
---------------------------
Test MammalTokenizer without requiring a GPU.
Requires a network connection on first run to download the tokenizer weights.
"""

import pytest

from examples.example_prompts import (
    GENE_BRCA1,
    PROTEIN_CALMODULIN,
    SMILES_ASPIRIN,
)
from vllm_mammal_plugin.tokenization import MammalTokenizer


class TestTokenization:
    """Test MammalTokenizer without requiring a GPU.

    The tokenizer is downloaded once per test class via a class-scoped fixture
    to avoid redundant downloads across tests in this class.
    """

    MODEL_NAME = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"

    @pytest.fixture(scope="class")
    @classmethod
    def tokenizer(cls) -> MammalTokenizer:
        return MammalTokenizer.from_pretrained(cls.MODEL_NAME)

    def test_from_pretrained_is_callable(self) -> None:
        """MammalTokenizer.from_pretrained must be a classmethod callable."""
        assert callable(MammalTokenizer.from_pretrained)

    def test_encode_returns_nonempty_int_list(self, tokenizer: MammalTokenizer) -> None:
        """encode() must return a non-empty list of ints for each modality."""
        tokens = {
            "protein": tokenizer.encode(PROTEIN_CALMODULIN),
            "smiles": tokenizer.encode(SMILES_ASPIRIN),
            "gene": tokenizer.encode(GENE_BRCA1),
        }
        for modality, ids in tokens.items():
            assert isinstance(ids, list) and len(ids) > 0, modality
            assert all(isinstance(t, int) for t in ids), modality

    def test_different_inputs_produce_different_tokens(
        self, tokenizer: MammalTokenizer
    ) -> None:
        """Different prompts must produce distinct token sequences."""
        protein_ids = tokenizer.encode(PROTEIN_CALMODULIN)
        smiles_ids = tokenizer.encode(SMILES_ASPIRIN)
        gene_ids = tokenizer.encode(GENE_BRCA1)

        assert protein_ids != smiles_ids
        assert smiles_ids != gene_ids

    def test_protocol_properties(self, tokenizer: MammalTokenizer) -> None:
        """MammalTokenizer must satisfy the vLLM TokenizerLike protocol."""
        assert isinstance(tokenizer.vocab_size, int) and tokenizer.vocab_size > 0
        assert isinstance(tokenizer.max_token_id, int) and tokenizer.max_token_id > 0
        assert isinstance(tokenizer.pad_token_id, int)
        assert isinstance(tokenizer.eos_token_id, int)
        assert tokenizer.is_fast is True
        assert tokenizer.truncation_side == "right"
        assert isinstance(tokenizer.all_special_tokens, list)
        assert len(tokenizer.all_special_tokens) > 0
        vocab = tokenizer.get_vocab()
        assert isinstance(vocab, dict) and len(vocab) > 0

    def test_call_returns_batch_encoding_with_input_ids(
        self, tokenizer: MammalTokenizer
    ) -> None:
        """__call__ must return a BatchEncoding dict with a non-empty 'input_ids' key."""
        encoding = tokenizer(PROTEIN_CALMODULIN)
        assert "input_ids" in encoding
        assert isinstance(encoding["input_ids"], list)
        assert len(encoding["input_ids"]) > 0


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

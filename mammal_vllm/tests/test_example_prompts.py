"""
tests/test_example_prompts.py
------------------------------
Sanity-check the shared example-prompt constants in ``examples/example_prompts.py``.

These constants are used as inputs by the tokenization and embedding tests as
well as by the usage examples.  A malformed constant (e.g. a missing tag or
dropped ``<EOS>``) would silently corrupt every test that depends on it, so
this file validates their structure cheaply with no GPU or network required.
"""

import pytest
from examples.example_prompts import (
    GENE_BRCA1,
    GENE_MALAT1,
    PROTEIN_CALMODULIN,
    PROTEIN_FLUORESCENT,
    SMILES_ASPIRIN,
    SMILES_CAFFEINE,
)


class TestExamplePromptsStructure:
    """Verify the structure of every shared example-prompt constant."""

    def test_protein_calmodulin(self) -> None:
        assert PROTEIN_CALMODULIN.startswith("<@TOKENIZER-TYPE=AA>")
        assert "<MOLECULAR_ENTITY_OF_TYPE_PROTEIN>" in PROTEIN_CALMODULIN
        assert PROTEIN_CALMODULIN.endswith("<EOS>")

    def test_protein_fluorescent(self) -> None:
        assert PROTEIN_FLUORESCENT.startswith("<@TOKENIZER-TYPE=AA>")
        assert "<MOLECULAR_ENTITY_OF_TYPE_PROTEIN>" in PROTEIN_FLUORESCENT
        assert PROTEIN_FLUORESCENT.endswith("<EOS>")

    def test_smiles_aspirin(self) -> None:
        assert SMILES_ASPIRIN.startswith("<@TOKENIZER-TYPE=SMILES>")
        assert "<MOLECULAR_ENTITY_OF_TYPE_SMALL_MOL>" in SMILES_ASPIRIN
        assert SMILES_ASPIRIN.endswith("<EOS>")

    def test_smiles_caffeine(self) -> None:
        assert SMILES_CAFFEINE.startswith("<@TOKENIZER-TYPE=SMILES>")
        assert "<MOLECULAR_ENTITY_OF_TYPE_SMALL_MOL>" in SMILES_CAFFEINE
        assert SMILES_CAFFEINE.endswith("<EOS>")

    def test_gene_brca1(self) -> None:
        assert GENE_BRCA1.startswith("<@TOKENIZER-TYPE=GENE>")
        assert "<MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>" in GENE_BRCA1
        assert GENE_BRCA1.endswith("<EOS>")

    def test_gene_malat1(self) -> None:
        assert GENE_MALAT1.startswith("<@TOKENIZER-TYPE=GENE>")
        assert "<MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>" in GENE_MALAT1
        assert GENE_MALAT1.endswith("<EOS>")


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

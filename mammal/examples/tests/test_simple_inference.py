import pytest
import torch
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

from mammal.keys import *
from mammal.model import Mammal


@pytest.fixture(scope="module")
def tmp_cache_dir(tmp_path_factory):
    model_dir_path = tmp_path_factory.mktemp("simple_infer")
    return model_dir_path


def test_simple_infer(tmp_cache_dir: str) -> None:
    """
    A simple function that test the proposed inference example on HF, https://huggingface.co/ibm/biomed.omics.bl.sm.ma-ted-458m
    """

    # Load Model
    model = Mammal.from_pretrained(
        "ibm/biomed.omics.bl.sm.ma-ted-458m", cache_dir=tmp_cache_dir
    )
    model.eval()

    # Load Tokenizer
    tokenizer_op = ModularTokenizerOp.from_pretrained(
        "ibm/biomed.omics.bl.sm.ma-ted-458m", cache_dir=tmp_cache_dir
    )

    # Prepare Input Prompt
    protein_calmodulin = "MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMISELDQDGFIDKEDLHDGDGKISFEEFLNLVNKEMTADVDGDGQVNYEEFVTMMTSK"
    protein_calcineurin = "MSSKLLLAGLDIERVLAEKNFYKEWDTWIIEAMNVGDEEVDRIKEFKEDEIFEEAKTLGTAEMQEYKKQKLEEAIEGAFDIFDKDGNGYISAAELRHVMTNLGEKLTDEEVDEMIRQMWDQNGDWDRIKELKFGEIKKLSAKDTRGTIFIKVFENLGTGVDSEYEDVSKYMLKHQ"

    # Create and load sample
    sample_dict = dict()
    # Formatting prompt to match pre-training syntax
    sample_dict[ENCODER_INPUTS_STR] = (
        f"<@TOKENIZER-TYPE=AA><BINDING_AFFINITY_CLASS><SENTINEL_ID_0><MOLECULAR_ENTITY><MOLECULAR_ENTITY_GENERAL_PROTEIN><SEQUENCE_NATURAL_START>{protein_calmodulin}<SEQUENCE_NATURAL_END><MOLECULAR_ENTITY><MOLECULAR_ENTITY_GENERAL_PROTEIN><SEQUENCE_NATURAL_START>{protein_calcineurin}<SEQUENCE_NATURAL_END><EOS>"
    )

    # Tokenize
    tokenizer_op(
        sample_dict=sample_dict,
        key_in=ENCODER_INPUTS_STR,
        key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
        key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK,
    )
    sample_dict[ENCODER_INPUTS_TOKENS] = torch.tensor(
        sample_dict[ENCODER_INPUTS_TOKENS]
    )
    sample_dict[ENCODER_INPUTS_ATTENTION_MASK] = torch.tensor(
        sample_dict[ENCODER_INPUTS_ATTENTION_MASK]
    )

    # Generate Prediction
    batch_dict = model.generate(
        [sample_dict],
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=5,
    )

    # Get output
    generated_output = tokenizer_op._tokenizer.decode(batch_dict[CLS_PRED][0])
    print(f"{generated_output=}")

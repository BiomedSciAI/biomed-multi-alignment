from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

from mammal.examples.scrna_mlm.pl_data_module import ScRNAMLMDataModule
from mammal.examples.scrna_mlm.task import ScRNAMLMTask
from mammal.keys import *

data_module_kwargs = {
    "data_path": "data/Zheng_68k_preprocessed.h5ad",  # this should be absolute or relative to the directory with the example code
    "batch_size": 20,
    "train_dl_kwargs": {"num_workers": 8},
    "valid_dl_kwargs": {"num_workers": 8},
    "input_max_seq_length": 20,
    "encoder_input_max_seq_len": 128,
}


def test_gene_expression_mlm():
    # Load Tokenizer
    modular_tokenizer = ModularTokenizerOp.from_pretrained(
        "/Users/yoavkt/.cache/huggingface/hub/models--ibm--biomed.omics.bl.sm.ma-ted-458m/snapshots/6d319d8dcf97f8821635327fc8cda24670553daa/tokenizer"
    )
    sct = ScRNAMLMTask(
        data_module_kwargs=data_module_kwargs, tokenizer_op=modular_tokenizer
    )
    dm = ScRNAMLMDataModule(
        **data_module_kwargs,
        tokenizer_op=modular_tokenizer,
        data_preprocessing=sct.data_preprocessing,
    )
    dm.setup(stage="train")
    train_dl = dm.train_dataloader()

    sample = next(iter(train_dl))
    assert sample is not None
    for index in range(len(sample[ENCODER_INPUTS_STR])):
        encoder_input = sample[ENCODER_INPUTS_STR][index]
        decoder_input = sample[DECODER_INPUTS_STR][index]
        lbl = sample[LABELS_STR][index]
        assert "<DECODER_START>" in decoder_input
        assert decoder_input.replace("<DECODER_START>", "") == lbl
        assert "SENTINEL_ID" in decoder_input  # encoder
        assert "SENTINEL_ID" in encoder_input
        assert (
            "<MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>"
            in encoder_input
        )
        assert "<EOS>" in encoder_input
        assert "<@TOKENIZER-TYPE=GENE>" in encoder_input

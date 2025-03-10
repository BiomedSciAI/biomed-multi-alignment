import functools
import re

from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

# from mammal.examples.scrna_cell_type.pl_data_module import CellTypeDataModule
# from mammal.examples.scrna_cell_type.task import CellTypeTask
from mammal.examples.scrna_mlm.pl_data_module import ScRNAMLMDataModule
from mammal.examples.scrna_mlm.task import ScRNAMLMTask

data_module_kwargs = {
    "data_path": "data/Zheng_68k_preprocessed.h5ad",  # this should be absolute or relative to the directory with the example code
    "batch_size": 20,
    # tokenizer_op is provided later, dymanicly
    "train_dl_kwargs": {"num_workers": 8},  # Dataloader constructor parameters
    "valid_dl_kwargs": {"num_workers": 8},  # Dataloader constructor parameters
    # data_preprocessing is provided later, dymanicly
    "input_max_seq_length": 500,
    "encoder_input_max_seq_len": 512,
    "labels_max_seq_len": 20,
}


def test_gene_expression_cell_type():
    # Load Tokenizer
    modular_tokenizer = ModularTokenizerOp.from_pretrained(
        "/Users/yoavkt/.cache/huggingface/hub/models--ibm--biomed.omics.bl.sm.ma-ted-458m/snapshots/6d319d8dcf97f8821635327fc8cda24670553daa/tokenizer"
    )
    """Validate task format."""
    dm = ScRNAMLMDataModule(
        **data_module_kwargs,
        tokenizer_op=modular_tokenizer,
        data_preprocessing=ScRNAMLMTask.data_preprocessing,
    )
    dm.setup(stage="train")
    train_dl = dm.train_dataloader()

    sample = next(iter(train_dl))
    assert sample is not None
    for index in range(len(sample["data.query.encoder_input"])):
        encoder_input = sample["data.query.encoder_input"][index]
        assert "<MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>" in encoder_input
        assert len(re.findall(r"\[(.*?)\]", encoder_input)) > 0
        # assert label_special_token + "<SENTINEL_ID_0>" in encoder_input
        assert "<EOS>" in encoder_input


def test_t5_cta_data_loader(modular_tokenizer, cta_data_module_kwargs_except_tokenizer):
    task = "t5_cta_zheng68k"
    task_partial = functools.partial(
        ScRNAMLMTask,  # tasks.T5CellTypeAnnotationBmfmTask,
        name=task,
        weight=1.0,
        loss="ce",
        data_module_kwargs=cta_data_module_kwargs_except_tokenizer,
    )
    # tasks_callables = {task: task_partial}
    # tasks_callables = {task: task_partial}
    task_obj = task_partial(clearml_logger=None, tokenizer_op=modular_tokenizer)
    task_obj.name = task
    """
    task_list = tasks.make_bmfm_task_list(
        tasks_callables=tasks_callables,
        clearml_logger=None,
        tokenizer=modular_tokenizer,
    )
    """
    # task_list = []
    data_model_kwargs = {
        "data_path": "./scrna_cell_type/data/Zheng_68k_preprocessed.h5ad",
        "batch_size": 20,
        "tokenizer_op": modular_tokenizer,
        "data_preprocessing": None,
        "train_dl_kwargs": {},
        "valid_dl_kwargs": {},
    }
    return data_model_kwargs
    # pl_data_module = main_train.data(task_list, seed=1, mb_per_epoch=24)
    # pl_data_module = ScRNADataModule(**data_model_kwargs)
    # pl_data_module.setup(stage="train")
    # train_dl = pl_data_module.train_dataloader()
    # sample = next(iter(train_dl))[task][0]

    # encoder_input = sample["data.query.encoder_input"][0]
    # labels = sample["data.query.labels"][0]

    # assert "<MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>" in encoder_input
    # assert "<SENTINEL_ID_0>" in encoder_input
    # assert "<SENTINEL_ID_0>" in labels
    # assert "[CL:" in labels

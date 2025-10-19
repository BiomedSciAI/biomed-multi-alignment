import socket
from pathlib import Path

import hydra
import pytest
import pytorch_lightning as pl
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from fuse.utils.multiprocessing.helpers import num_available_cores
from omegaconf import OmegaConf

from mammal.main_finetune import *

# pylint: disable=W0621

"""_summary_

Testing examples
"""

TEST_CONFIG_DIRPATH = str(Path(__file__).parents[0] / "../protein_solubility")
TEST_CONFIG_FILENAME = "config.yaml"

STATIC_OVERRIDES = [
    "track_clearml=null",  # Travis cannot connect to ClearML at the moment. We might be able to fix it with a dedicated user + config credentials.
    "trainer.max_epochs=1",  # Small number for a faster run.
    "+trainer.limit_train_batches=2",  # Small number for a faster run.
    "+trainer.limit_val_batches=3",  # Small number for a faster run.
    "+trainer.enable_checkpointing=False",  # Do not checkpoint - saves memory
    "model_dir=null",
]

OVERRIDES = [
    "task.data_module_kwargs.train_dl_kwargs.num_workers=0",  # Using parallelization cause co
    "task.data_module_kwargs.valid_dl_kwargs.num_workers=0",  # Using parallelization cause co
    "root=.",
    "name=sol_test",
    "+tokenizer.new_special_tokens=['<my_special_token1>','special_token2',yet_another_special_token]",
] + STATIC_OVERRIDES


@pytest.fixture(scope="module")
def cfg_dict():
    with hydra.initialize_config_dir(TEST_CONFIG_DIRPATH, version_base="1.1"):
        _cfg = hydra.compose(TEST_CONFIG_FILENAME, overrides=OVERRIDES)
        yield _cfg


@pytest.fixture(scope="module")
def cfg_obj(cfg_dict):
    OmegaConf.register_new_resolver("num_cores_auto", num_available_cores, replace=True)
    cfg_obj = hydra.utils.instantiate(cfg_dict)
    return cfg_obj


@pytest.fixture(scope="module")
def cfg(cfg_obj):
    return cfg_obj


@pytest.fixture(scope="module")
def clearml_logger():
    return None


def test_context(cfg_dict):
    assert cfg_dict


def test_context_obj(cfg):
    assert cfg


def seed(seed_value: int) -> int:
    pl.seed_everything(seed_value, workers=True)

    return seed_value


def test_seed():
    original_seed_value = 12345
    seed_value = seed(original_seed_value)
    assert seed_value == original_seed_value


@pytest.fixture(scope="module")
def tokenizer_op(cfg_dict):
    # return ModularTokenizerOp.from_pretrained(cfg_dict.tokenizer.tokenizer_path)
    # The tokenizer is loaded with the extra special tokens, so we can check they are avaible
    return load_and_update_tokenizer_op(cfg_dict)


def test_tokenizer(tokenizer_op):
    assert isinstance(tokenizer_op, ModularTokenizerOp)
    special_tokens = [
        "<MOLECULAR_ENTITY>",
        "<MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>",
        "<CELL_TYPE_CLASS>",
        "<PAD>",
        "<EOS>",
        "<DECODER_START>",
        "<SENTINEL_ID_0>",
        "<my_special_token1>",
        "special_token2",
        "yet_another_special_token",
    ]
    for token in special_tokens:
        tokenizer_op.get_token_id(token)  # throws assert if fails
    never_seen_before = "never_seen_before"
    with pytest.raises(AssertionError):
        tokenizer_op.get_token_id(never_seen_before)
    num_new = tokenizer_op.add_new_special_tokens(
        [
            never_seen_before,
        ]
    )
    assert num_new == 1
    tokenizer_op.get_token_id(never_seen_before)


@pytest.fixture(scope="module")
def current_train_session_metadata():
    return {}


@pytest.fixture(scope="module")
def test_task(cfg_obj, clearml_logger, tokenizer_op):
    _task_list = cfg_obj.task(
        tokenizer_op=tokenizer_op,
        logger=clearml_logger,
    )
    return _task_list


@pytest.fixture(scope="module")
def pl_data_module(test_task):
    """get lightning data module"""
    return test_task.data_module()


@pytest.fixture(scope="module")
def pl_module(test_task, cfg):
    """get lightning module"""
    model = Mammal.from_pretrained(
        cfg.model.pretrained_kwargs.pretrained_model_name_or_path
    )
    _pl_module = module(
        model=model,
        task=test_task,
        **OmegaConf.to_container(cfg.module, resolve=True),
    )
    return _pl_module


@pytest.mark.skipif(
    "ccc" not in socket.gethostname(),
    reason="Train consumes too much memory for a Travis run.",
)
def test_evaluate(cfg, pl_data_module, pl_module):
    pl_trainer = pl.Trainer(**cfg.trainer)

    pl_data_module.setup("test")
    out = pl_trainer.validate(
        model=pl_module,
        dataloaders=pl_data_module.test_dataloader(),
    )
    print(out)


@pytest.mark.skipif(
    "ccc" not in socket.gethostname(),
    reason="Train consumes too much memory for a Travis run.",
)
def test_train(cfg, pl_data_module, pl_module):
    pl_trainer = pl.Trainer(**cfg.trainer)
    pl_trainer.fit(model=pl_module, datamodule=pl_data_module)

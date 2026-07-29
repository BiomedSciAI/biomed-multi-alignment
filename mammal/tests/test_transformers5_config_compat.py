"""
Regression tests for the transformers>=5 migration.

`T5Config.__post_init__` forces `tie_word_embeddings = True` and folds the stored value
into a new `scale_decoder_outputs` attribute. MAMMAL has an *untied* `lm_head`. Unpickling
skips `__post_init__` entirely, so a config from a transformers 4 era `.ckpt` is missing
every field transformers 5 added.
"""

import copy
import json

import pytest
import torch
from transformers import T5Config

from mammal.model import Mammal, MammalConfig

# Shape-only stand-in for the published ma-ted-458m config; `tie_word_embeddings: False`
# and `transformers_version: 4.43.4` are what that config.json actually carries.
CONFIG_DICT = {
    "support_input_scalars": True,
    "transformers_version": "4.43.4",
    "t5_config": {
        "d_ff": 64,
        "d_kv": 8,
        "d_model": 32,
        "decoder_start_token_id": 16,
        "eos_token_id": 5,
        "feed_forward_proj": "gated-gelu",
        "num_decoder_layers": 1,
        "num_heads": 4,
        "num_layers": 1,
        "pad_token_id": 1,
        "tie_word_embeddings": False,
        "vocab_size": 64,
    },
}

# On a transformers 5 `T5Config.__dict__` but not a 4.43 one. `torch_dtype` was renamed
# `dtype`; `scale_decoder_outputs` is derived rather than inherited.
V5_ONLY = [
    "_experts_implementation_internal",
    "_output_attentions",
    "dtype",
    "scale_decoder_outputs",
]


def _tf4_config(*missing: str) -> MammalConfig:
    """A `MammalConfig` shaped as one unpickled from a 4.43 `.ckpt` would be."""
    config = copy.deepcopy(MammalConfig.from_dict(copy.deepcopy(CONFIG_DICT)))
    for attr in missing:
        config.t5_config.__dict__.pop(attr, None)
    config.t5_config.__dict__["torch_dtype"] = None  # the 4.x spelling
    return config


def test_lm_head_stays_untied_however_the_config_was_built():
    """
    A config built directly must give the same untied `lm_head` as one from `from_dict`,
    where the restore lives. Otherwise it silently gets a different architecture.
    """
    config = MammalConfig(
        t5_config=T5Config.from_dict(copy.deepcopy(CONFIG_DICT["t5_config"])),
        support_input_scalars=True,
    )
    t5 = Mammal(config).t5_model

    assert t5.lm_head.weight.data_ptr() != t5.shared.weight.data_ptr()


def test_missing_tie_flag_in_config_json_is_not_silently_accepted(tmp_path):
    """
    Without the key the model is built tied. Loading a published-format checkpoint with
    `strict=False` then succeeds while `get_input_embeddings()` stays randomly initialised.
    """
    reference = Mammal(MammalConfig.from_dict(copy.deepcopy(CONFIG_DICT)))
    reference._save_pretrained(tmp_path)

    config_path = tmp_path / "config.json"
    saved = json.loads(config_path.read_text())
    del saved["t5_config"]["tie_word_embeddings"]
    config_path.write_text(json.dumps(saved))

    loaded = Mammal.from_pretrained(str(tmp_path), strict=False)

    assert torch.equal(
        loaded.t5_model.get_input_embeddings().weight,
        reference.t5_model.get_input_embeddings().weight,
    )


@pytest.mark.parametrize("missing", V5_ONLY)
def test_each_missing_v5_attr_is_backfilled(missing):
    """Which attribute is missing must not decide whether a checkpoint loads."""
    model = Mammal(_tf4_config(missing))
    model.t5_model(
        input_ids=torch.tensor([[3, 4, 5]]),
        decoder_input_ids=torch.tensor([[16]]),
    )


def test_legacy_ckpt_loads_runs_and_resaves(tmp_path):
    """
    `from_pretrained` falls back to `pl_ckpt_dict["config"]` when no `config.json` sits
    next to the `.ckpt` (the README's `evaluate=True ... best_epoch.ckpt` flows). Passed
    as a `Path`, which the signature advertises.
    """
    reference = Mammal(MammalConfig.from_dict(copy.deepcopy(CONFIG_DICT)))
    path = tmp_path / "best_epoch.ckpt"
    torch.save(
        {
            "config": _tf4_config(*V5_ONLY),
            "state_dict": {f"_model.{k}": v for k, v in reference.state_dict().items()},
        },
        path,
    )

    loaded = Mammal.from_pretrained(path)
    loaded.t5_model(
        input_ids=torch.tensor([[3, 4, 5]]),
        decoder_input_ids=torch.tensor([[16]]),
    )
    assert torch.equal(
        loaded.t5_model.lm_head.weight, reference.t5_model.lm_head.weight
    )

    out_dir = tmp_path / "resaved"
    out_dir.mkdir()
    loaded._save_pretrained(out_dir)
    resaved = json.loads((out_dir / "config.json").read_text())["t5_config"]

    assert resaved["tie_word_embeddings"] is False
    assert resaved["scale_decoder_outputs"] is False


def test_from_dict_does_not_mutate_the_caller_dict():
    """`from_dict` replaces `config_dict["t5_config"]` with a `T5Config` object."""
    config_dict = copy.deepcopy(CONFIG_DICT)

    MammalConfig.from_dict(config_dict)

    assert isinstance(config_dict["t5_config"], dict)
    MammalConfig.from_dict(config_dict)  # must be callable twice

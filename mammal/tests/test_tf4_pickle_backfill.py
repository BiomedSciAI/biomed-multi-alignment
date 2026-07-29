"""
Failing tests for gaps left by the transformers>=5 migration.

`from_pretrained` falls back to `pl_ckpt_dict["config"]` for a `.ckpt` with no adjacent
`config.json`, unpickling a config object written by transformers 4. Unpickling runs
neither `__init__` nor `__post_init__`, so every attribute transformers 5 added is absent
and the first v5 code path to dereference one raises. `Mammal.__init__` back-fills one
such attribute; there are four.
"""

import copy

import pytest
import torch

from mammal.model import Mammal, MammalConfig

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

# On a transformers 5 `T5Config.__dict__` but not a 4.43 one, so no transformers 4 pickle
# can carry them. `torch_dtype` was renamed `dtype`, hence the pair.
V5_ONLY = ["_experts_implementation_internal", "_output_attentions", "dtype"]
V5_ONLY_ALREADY_HANDLED = ["scale_decoder_outputs"]


def _tf4_config(*missing: str) -> MammalConfig:
    """A `MammalConfig` shaped as one unpickled from a 4.43 `.ckpt` would be."""
    config = copy.deepcopy(MammalConfig.from_dict(copy.deepcopy(CONFIG_DICT)))
    for attr in missing:
        config.t5_config.__dict__.pop(attr, None)
    config.t5_config.__dict__["torch_dtype"] = None  # the 4.x spelling
    return config


@pytest.mark.parametrize("missing", V5_ONLY + V5_ONLY_ALREADY_HANDLED)
def test_each_missing_v5_attr_is_backfilled(missing):
    """
    Which attribute happens to be missing should not decide whether a checkpoint loads.
    Only `scale_decoder_outputs` is handled today; the other three each fail alone,
    which is the argument for a generic back-fill over `PretrainedConfig().__dict__`
    rather than one `hasattr` check.
    """
    model = Mammal(_tf4_config(missing))
    model.t5_model(
        input_ids=torch.tensor([[3, 4, 5]]),
        decoder_input_ids=torch.tensor([[16]]),
    )


def test_legacy_ckpt_on_disk_loads(tmp_path):
    """
    End-to-end over the real `.ckpt` branch with every v5-only attribute absent at once,
    asserting the `lm_head` weights actually arrived rather than just that nothing threw.
    """
    reference = Mammal(MammalConfig.from_dict(copy.deepcopy(CONFIG_DICT)))
    path = tmp_path / "best_epoch.ckpt"
    torch.save(
        {
            "config": _tf4_config(*V5_ONLY, *V5_ONLY_ALREADY_HANDLED),
            "state_dict": {f"_model.{k}": v for k, v in reference.state_dict().items()},
        },
        path,
    )

    loaded = Mammal.from_pretrained(path)

    assert torch.equal(
        loaded.t5_model.lm_head.weight, reference.t5_model.lm_head.weight
    )


def test_from_dict_does_not_mutate_the_caller_dict():
    """
    `from_dict` writes a `T5Config` back into `config_dict["t5_config"]` and only
    deepcopies when `allow_config_mismatch=True`, so the caller's dict returns with an
    object where it had a dict and any second use of it breaks.
    """
    config_dict = copy.deepcopy(CONFIG_DICT)

    MammalConfig.from_dict(config_dict)

    assert isinstance(config_dict["t5_config"], dict), "caller's dict was mutated"
    MammalConfig.from_dict(config_dict)  # must be callable twice

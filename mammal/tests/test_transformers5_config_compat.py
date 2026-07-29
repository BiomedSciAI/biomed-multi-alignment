"""
Regression tests for the transformers>=5 migration.

`T5Config.__post_init__` forces `tie_word_embeddings = True` and folds the stored value
into a new `scale_decoder_outputs` attribute. MAMMAL has an *untied* `lm_head`, and with
`tie_word_embeddings` False transformers>=5 ties nothing at all - including
`encoder/decoder.embed_tokens <-> shared`. These tests pin both invariants where the
code relies on them (`Mammal.__init__` / `from_pretrained`) rather than where they
happen to be restored (`MammalConfig.from_dict`).
"""

import copy
import json

import torch
from transformers import T5Config

from mammal.model import Mammal, MammalConfig

# Shape-only stand-in for the published ma-ted-458m config; `tie_word_embeddings: False`
# and `transformers_version: 4.43.4` are what that config.json actually carries, i.e.
# every checkpoint in the wild was written by transformers 4.
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


def _legacy_ckpt(tmp_path):
    """A Lightning `.ckpt` whose pickled config predates `scale_decoder_outputs`."""
    model = Mammal(MammalConfig.from_dict(copy.deepcopy(CONFIG_DICT)))
    config = copy.deepcopy(model.config)
    # transformers 4 never set this; unpickling does not run `__post_init__`.
    config.t5_config.__dict__.pop("scale_decoder_outputs", None)

    path = tmp_path / "best_epoch.ckpt"
    state_dict = {f"_model.{k}": v for k, v in model.state_dict().items()}
    torch.save({"config": config, "state_dict": state_dict}, path)
    return path


def test_lm_head_stays_untied_however_the_config_was_built():
    """
    A `MammalConfig` built directly - a from-scratch/hydra training config, or one
    assembled in code - must give the same untied `lm_head` as one from `from_dict`,
    where the restore currently lives. Otherwise it silently gets a different
    architecture: `lm_head` aliased onto the input embedding table, no error.
    """
    config = MammalConfig(
        t5_config=T5Config.from_dict(copy.deepcopy(CONFIG_DICT["t5_config"])),
        support_input_scalars=True,
    )
    t5 = Mammal(config).t5_model
    tied = t5.lm_head.weight.data_ptr() == t5.shared.weight.data_ptr()

    assert not tied, "lm_head was silently tied to the input embeddings"


def test_missing_tie_flag_in_config_json_is_not_silently_accepted(tmp_path):
    """
    `from_dict` only restores the flag `if "tie_word_embeddings" in t5_config_dict`, so
    without the key the model is built tied. Loading a published-format checkpoint into
    it with `strict=False` then succeeds while `get_input_embeddings()` - the table
    `_calculate_inputs_embeddings` reads - stays at its random initialisation.
    """
    reference = Mammal(MammalConfig.from_dict(copy.deepcopy(CONFIG_DICT)))
    reference._save_pretrained(tmp_path)

    config_path = tmp_path / "config.json"
    saved = json.loads(config_path.read_text())
    del saved["t5_config"]["tie_word_embeddings"]
    config_path.write_text(json.dumps(saved))

    loaded = Mammal.from_pretrained(str(tmp_path), strict=False)
    got_weights = torch.equal(
        loaded.t5_model.get_input_embeddings().weight,
        reference.t5_model.get_input_embeddings().weight,
    )

    assert got_weights, "loaded silently, but get_input_embeddings() is still random"


def test_config_pickled_by_transformers_4_runs_and_resaves(tmp_path):
    """
    `from_pretrained` falls back to `pl_ckpt_dict["config"]` when no `config.json` sits
    next to the `.ckpt` (the README's `evaluate=True ... best_epoch.ckpt` flows). That
    config has no `scale_decoder_outputs`, which `T5ForConditionalGeneration.forward`
    dereferences unconditionally. Passed as a `Path`, which the signature advertises.
    """
    loaded = Mammal.from_pretrained(_legacy_ckpt(tmp_path))

    loaded.t5_model(
        input_ids=torch.tensor([[3, 4, 5]]),
        decoder_input_ids=torch.tensor([[16]]),
    )

    out_dir = tmp_path / "resaved"
    out_dir.mkdir()
    loaded._save_pretrained(out_dir)
    resaved = json.loads((out_dir / "config.json").read_text())["t5_config"]

    assert resaved["tie_word_embeddings"] is False
    assert resaved["scale_decoder_outputs"] is False

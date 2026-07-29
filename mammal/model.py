import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from fuse.data.utils.collates import CollateDefault
from fuse.dl.models.heads.common import ClassifierMLP
from huggingface_hub import ModelHubMixin, snapshot_download
from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, load_model, save_model
from transformers import PretrainedConfig, T5Config, T5ForConditionalGeneration

from mammal.keys import *  # noqa
from mammal.lora import get_lora_model


def backfill_config_defaults(config: PretrainedConfig | None) -> None:
    """
    Restore attributes that transformers>=5 added to `PretrainedConfig` onto a config
    that never ran `__init__`/`__post_init__` - e.g. one unpickled from a transformers 4
    era checkpoint, which is missing them all.
    """
    if config is None:
        return
    state = config.__dict__
    # Renamed in transformers>=5. Mapped before the defaults so stored values win.
    state.setdefault("_output_attentions", state.get("output_attentions", None))
    state.setdefault("dtype", state.pop("torch_dtype", None))
    for key, value in PretrainedConfig().__dict__.items():
        state.setdefault(key, value)


@dataclass
class MammalConfig(PretrainedConfig):
    """
    Mammal Configuration
    """

    t5_config: T5Config = None  # base T5 model configuration
    transformers_version: str | None = (
        None  # Need it to be compatible with 'PretrainedConfig.from_dict()' method
    )
    use_lora: bool = (
        False  # whether to apply Lora or not - might be used for finetunning
    )
    encoder_head_layers: list[int] | None = None  # list of encoder head layer sizes
    encoder_head_dropout: float = (
        0.0  # the dropout to apply to each layer in encoder head except the last layer
    )
    auxiliary_encoder_heads: dict | None = (
        None  # not used - kept for backward compatibility
    )
    scalars_prediction_head: dict | None = None  # scalar prediction head configuration
    support_input_scalars: bool = False  # consider changing to default True soon
    random_weights: bool = False  # If True, will not load the pre-trained weights

    @classmethod
    def from_dict(
        cls, config_dict: dict[str, Any], *, allow_config_mismatch: bool = False
    ) -> "MammalConfig":
        if "t5_config" not in config_dict:
            raise ValueError(f"config_dict should have key 't5_config'. {config_dict=}")

        # We don't want to change the incoming dict - `t5_config` is replaced with a
        # `T5Config` object below, which would otherwise leave the caller holding an
        # object where it had a dict and break any second use of it.
        config_dict = copy.deepcopy(config_dict)

        if allow_config_mismatch:
            # Allowing to load the model even if the incoming config dict has unexpected key(s)
            mismatch_keys = []
            for incoming_config_key in list(config_dict.keys()):
                if incoming_config_key not in cls.__dataclass_fields__:
                    # Incoming key isn't part of the expected config keys
                    mismatch_keys.append(incoming_config_key)
                    config_dict.pop(incoming_config_key)

            if len(mismatch_keys) > 0:
                print(
                    f"Warning, mismatch detected! Make sure you know what you are doing... {mismatch_keys=}"
                )

        # We want to instantiate each class from it's dict (json), using the parent class logic
        # HF don't support the case where there are nested *different* configs.
        t5_config_dict = config_dict["t5_config"]
        t5_config = T5Config.from_dict(t5_config_dict)
        # transformers>=5 ignores `tie_word_embeddings` when it is supplied through
        # `from_dict`/kwargs and always falls back to the class default (True). This
        # model has an *untied* LM head (`lm_head` differs from the input embeddings),
        # so restore the stored value explicitly - otherwise the model ties `lm_head`
        # to `shared`, overwriting the real LM-head weights and corrupting generation.
        if "tie_word_embeddings" in t5_config_dict:
            t5_config.tie_word_embeddings = t5_config_dict["tie_word_embeddings"]
        config_dict["t5_config"] = t5_config
        config = cls(**config_dict)
        return config

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Unpickling hook - the `.ckpt` path in `from_pretrained` restores configs here."""
        self.__dict__.update(state)
        backfill_config_defaults(self)
        backfill_config_defaults(self.__dict__.get("t5_config"))

    @classmethod
    def from_path(
        cls, config_filepath: str, *, allow_config_mismatch: bool = False
    ) -> "MammalConfig":

        with open(config_filepath, encoding="utf-8") as f:
            config_dict = json.load(f)
        config = MammalConfig.from_dict(
            config_dict, allow_config_mismatch=allow_config_mismatch
        )

        return config

    @classmethod
    def get_deprecated_arguments(cls) -> list[str]:
        """Property of deprecated arguments to support backward compatibility."""
        deprecated_arguments = ["load_weights", "t5_pretrained_name"]
        return deprecated_arguments

    def override(self, config_overrides: dict[str, Any]) -> None:
        """
        Override existing (loaded configuration)
        """
        for key, value in config_overrides.items():
            assert hasattr(
                self, key
            ), f"Unexpected override, expecting fields from {self.__class__.__name__}"
            if isinstance(value, DictConfig):
                value = OmegaConf.to_container(value, resolve=True)
            if isinstance(value, dict) and getattr(self, key, None) is not None:
                for key_inner, value_inner in value.items():
                    if isinstance(getattr(self, key), dict):
                        getattr(self, key)[key_inner] = value_inner
                    else:
                        if not hasattr(getattr(self, key), key_inner):
                            print(
                                f"Warning: unexpected overrides, expecting to override fields already exist in {getattr(self, key).__class__.__name__} - adding new fields instead"
                            )
                        print(
                            f"{self.__class__.__name__}: overriding {key}.{key_inner} to {value_inner}"
                        )
                        setattr(getattr(self, key), key_inner, value_inner)
            else:
                print(f"{self.__class__.__name__}: overriding {key} to {value}")
                setattr(self, key, value)


class Mammal(ModelHubMixin, torch.nn.Module):
    VERSION = "0.0"

    def __init__(
        self,
        config: MammalConfig,
    ) -> None:
        """
        Create a Model without loading weights
        """
        super().__init__()
        self.config = config

        t5_config = self.config.t5_config
        # Also back-fill here, not just in `__setstate__`: a config assembled in code
        # likewise skips `__post_init__`. `forward` dereferences several of these fields.
        backfill_config_defaults(t5_config)
        # Derived by `T5Config.__post_init__`, so the generic back-fill can't supply it.
        if not hasattr(t5_config, "scale_decoder_outputs"):
            t5_config.scale_decoder_outputs = (
                getattr(t5_config, "tie_word_embeddings", False) is not False
            )
        # `__post_init__` forces this True; re-assert the untied `lm_head` where it's
        # relied on rather than trusting how the config was built.
        t5_config.tie_word_embeddings = False

        self.t5_model = T5ForConditionalGeneration(config=t5_config)

        # transformers>=5 refactored T5 so the encoder/decoder input-embedding
        # tables are shared with `shared` only when tie_word_embeddings is True.
        # This model keeps `lm_head` untied (tie_word_embeddings=False) but still
        # needs the encoder/decoder inputs to use the single shared embedding
        # table (which `get_input_embeddings()` returns and which the checkpoint
        # only stores once). Re-establish that sharing explicitly; on transformers
        # 4.x the encoder/decoder already share `shared`, so this is a no-op.
        self.t5_model.set_input_embeddings(self.t5_model.get_input_embeddings())

        if getattr(self.config, "support_input_scalars", False):
            self.project_input_scalars = torch.nn.Linear(
                1, self.t5_model.get_input_embeddings().embedding_dim, bias=True
            )
        else:
            self.project_input_scalars = None

        if self.config.encoder_head_layers is not None:
            self.encoder_head = get_encoder_mlp_head(
                num_classes=self.config.t5_config.vocab_size,
                layers=self.config.encoder_head_layers,
                dropout=self.config.encoder_head_dropout,
                embedding_size=self.t5_model.get_input_embeddings().embedding_dim,
            )
        if getattr(self.config, "scalars_prediction_head", None) is not None:
            num_classes = self.config.scalars_prediction_head.get("num_classes", 1)
            if self.config.scalars_prediction_head.get("predict_error", False):
                num_classes *= 2

            self.scalars_prediction_head = get_encoder_mlp_head(
                num_classes=num_classes,
                layers=self.config.scalars_prediction_head["layers"],
                dropout=self.config.scalars_prediction_head["dropout"],
                embedding_size=self.t5_model.get_input_embeddings().embedding_dim,
            )
        else:
            self.scalars_prediction_head = None

    def generate(self, samples: list | dict, **generate_kwargs: dict):
        """
        Generate, unlike forward_encoder_decoder which use teacher forcing. No need for decoder_input here.
        :params samples: batch_dict or list of sample_dict
        :params generate_kwargs: arguments to transformers.generation.GenerationMixin.generate() function
        """
        if isinstance(samples, dict):
            batch_dict = samples
        else:
            batch_dict = CollateDefault()(samples)

        input_embeddings = self._calculate_inputs_embeddings(batch_dict)

        generated_output = self.t5_model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=batch_dict[ENCODER_INPUTS_ATTENTION_MASK],
            eos_token_id=self.config.t5_config.eos_token_id,
            pad_token_id=self.config.t5_config.pad_token_id,
            **generate_kwargs,
        )

        # transformers>=5 removed the legacy per-strategy output aliases
        # (BeamSearch/GreedySearch/Sample/BeamSample*EncoderDecoderOutput). They
        # collapse into two encoder-decoder classes, which also exist in
        # transformers 4.x, so this stays compatible with both.
        MODEL_OUTPUT_SEARCH_TYPES = (
            transformers.generation.utils.GenerateEncoderDecoderOutput,  # greedy + sample
            transformers.generation.utils.GenerateBeamEncoderDecoderOutput,  # beam + beam-sample
        )

        # depending generate_kwargs, different types can be returned from model.generate(...)
        if isinstance(generated_output, MODEL_OUTPUT_SEARCH_TYPES):
            cls_pred = generated_output.sequences
            if "sequences_scores" in generated_output:
                batch_dict["model.out.sequences_beam_search_scores"] = generated_output[
                    "sequences_scores"
                ]
        elif isinstance(generated_output, torch.Tensor):
            cls_pred = generated_output
        else:
            raise Exception(f"unexpected return type {type(generated_output)}")

        cls_pred = cls_pred[:, 1:]
        if LABELS_TOKENS in batch_dict:
            labels = batch_dict[LABELS_TOKENS]
            if cls_pred.shape[-1] > labels.shape[-1]:  # truncate
                print(
                    f"warning: had to truncate generated sequences from  {cls_pred.shape[-1]} to {labels.shape[-1]} tokens"
                )
                cls_pred = cls_pred[:, : labels.shape[-1]]
            elif cls_pred.shape[-1] < labels.shape[-1]:  # pad
                cls_pred = torch.nn.functional.pad(
                    cls_pred,
                    (0, labels.shape[-1] - cls_pred.shape[-1]),
                    value=self.config.t5_config.pad_token_id,
                )

        batch_dict[CLS_PRED] = cls_pred.contiguous()
        if isinstance(generated_output, MODEL_OUTPUT_SEARCH_TYPES) and hasattr(
            generated_output, "scores"
        ):
            batch_dict[LOGITS] = torch.vstack(
                [x[None] for x in generated_output.scores]
            ).permute(1, 0, 2)
            batch_dict[SCORES] = torch.nn.functional.softmax(batch_dict[LOGITS], dim=-1)
        else:
            batch_dict[LOGITS] = None
            batch_dict[SCORES] = None

        batch_dict[CE_LOSS] = None
        return batch_dict

    def forward_encoder_only(self, samples: list | dict):
        """
        Use the encoder followed by an encoder head and scalar head. Should be use when predicting scalars.
        No need for decoder_input.
        :params samples: batch_dict or list of sample_dict
        """
        assert self.encoder_head is not None

        if isinstance(samples, dict):
            batch_dict = samples
        else:
            batch_dict = CollateDefault()(samples)

        input_embeddings = self._calculate_inputs_embeddings(batch_dict)

        model_out = self.t5_model.encoder(
            inputs_embeds=input_embeddings,
            attention_mask=batch_dict[ENCODER_INPUTS_ATTENTION_MASK],
        )

        batch_dict[LOGITS] = self.encoder_head(x=model_out["last_hidden_state"])
        batch_dict[SCORES] = torch.nn.functional.softmax(batch_dict[LOGITS], dim=-1)
        batch_dict[CLS_PRED] = batch_dict[SCORES].argmax(-1)
        batch_dict[ENCODER_LAST_HIDDEN_STATE] = model_out["last_hidden_state"]

        if self.scalars_prediction_head is not None:
            batch_dict[SCALARS_PREDICTION_HEAD_LOGITS] = self.scalars_prediction_head(
                model_out["last_hidden_state"]
            ).squeeze(dim=2)

        return batch_dict

    def forward_encoder_decoder(
        self,
        samples: list | dict,
    ):
        """
        :params samples: batch_dict or list of sample_dict
        """

        if isinstance(samples, dict):
            batch_dict = samples
        else:
            batch_dict = CollateDefault()(samples)

        input_embeddings = self._calculate_inputs_embeddings(batch_dict)

        model_out = self.t5_model.forward(
            inputs_embeds=input_embeddings,
            attention_mask=batch_dict[ENCODER_INPUTS_ATTENTION_MASK],
            labels=batch_dict.get(LABELS_TOKENS, None),
            decoder_input_ids=batch_dict.get(DECODER_INPUTS_TOKENS, None),
            decoder_attention_mask=batch_dict.get(DECODER_INPUTS_ATTENTION_MASK, None),
        )

        # perform softmax and argmax
        model_out = dict(model_out)

        model_out["scores"] = torch.nn.functional.softmax(model_out["logits"], dim=-1)
        model_out["cls_pred"] = model_out["scores"].argmax(-1)

        batch_dict["model.out"] = model_out

        return batch_dict

    def forward(self, batch_dict: dict) -> dict:
        """
        Forward pass,
        Will dispatched according to forward_mode that specified in batch_dict
        """
        forward_mode = batch_dict.get("forward_mode", "forward")
        assert forward_mode in [
            "forward",
            "generate",
            "encoder",
            "enc_only_with_dec_format",
        ]

        if forward_mode == "forward":
            batch_dict = self.forward_encoder_decoder(batch_dict)

        elif forward_mode == "generate":
            generate_kwargs = batch_dict.get("generate_kwargs", {})
            batch_dict = self.generate(batch_dict, **generate_kwargs)

        elif forward_mode == "encoder":
            batch_dict = self.forward_encoder_only(batch_dict)

        return batch_dict

    def _calculate_inputs_embeddings(self, batch_dict: dict) -> torch.Tensor:
        """
        prepare input embeddings for the t5 model
        """
        inputs_embeds = self.t5_model.get_input_embeddings()(
            batch_dict[ENCODER_INPUTS_TOKENS]
        )

        if self.project_input_scalars is not None:
            if (ENCODER_INPUTS_SCALARS_VALUES in batch_dict) and (
                batch_dict[ENCODER_INPUTS_SCALARS_VALUES] is not None
            ):

                assert (
                    ENCODER_INPUTS_SCALARS_VALID_MASK in batch_dict
                ), f"expected ENCODER_INPUTS_SCALARS_VALID_MASK={ENCODER_INPUTS_SCALARS_VALID_MASK} to be found, since ENCODER_INPUTS_SCALARS_VALUES={ENCODER_INPUTS_SCALARS_VALUES} was found"
                assert (
                    batch_dict[ENCODER_INPUTS_SCALARS_VALID_MASK] is not None
                ), f"expected ENCODER_INPUTS_SCALARS_VALID_MASK={ENCODER_INPUTS_SCALARS_VALID_MASK} to not point to None, since ENCODER_INPUTS_SCALARS_VALUES={ENCODER_INPUTS_SCALARS_VALUES} was found and was not None"

                encoder_input_scalars_values = batch_dict[ENCODER_INPUTS_SCALARS_VALUES]
                encoder_input_scalars_valid_mask = batch_dict[
                    ENCODER_INPUTS_SCALARS_VALID_MASK
                ]

                if (
                    encoder_input_scalars_valid_mask.any().item()
                ):  # any input scalars used
                    projected = self.project_input_scalars(
                        encoder_input_scalars_values[..., None]
                    )  # each single scalar gets projected to model dim (e.g. 768)
                    inputs_embeds[encoder_input_scalars_valid_mask] += projected[
                        encoder_input_scalars_valid_mask
                    ]

        encoder_input_add_embeddings = batch_dict.get(
            ENCODER_INPUT_ADD_EMBEDDINGS, None
        )
        if encoder_input_add_embeddings is not None:
            inputs_embeds += encoder_input_add_embeddings

        return inputs_embeds

    def _save_pretrained(
        self,
        save_directory: Path,
        save_config_only: bool = False,
    ) -> None:
        print(f"Saving @ {save_directory}")

        # Define paths
        config_json_path = os.path.join(save_directory, CONFIG_NAME)
        self.config.to_json_file(json_file_path=config_json_path)

        if not save_config_only:
            model_safetensors_path = os.path.join(
                save_directory, SAFETENSORS_SINGLE_FILE
            )
            save_model(self, model_safetensors_path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        allow_config_mismatch: bool = False,
        config: MammalConfig | str | os.PathLike | None = None,
        config_overrides: dict[str, Any] | None = None,
        strict: bool = True,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> "Mammal":
        """
        :param pretrained_model_name_or_path: might be:
                                              (1) huggingface repo id
                                              (2) path to '.ckpt' file. We assume that the parent folder contains config.json file for the model
                                              (3) path to directory that contains 'model.safetensors' and 'config.json'
        :param config: typically not need to configure
        :param config_overrides: load config, but override specific fields from mammal MammalConfig.
                                  The dictionary should only include the fields to override.
        """
        # `Path` is accepted per the signature, but the checks below are string based.
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        if not os.path.exists(pretrained_model_name_or_path):
            print(
                f"Path doesn't exist. Will try to download from hf hub. {pretrained_model_name_or_path=}"
            )
            # Download ckpt dir from HF
            try:
                pretrained_model_name_or_path = snapshot_download(
                    repo_id=str(pretrained_model_name_or_path),
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except Exception as e:
                raise Exception(
                    f"Couldn't find the checkpoint path nor download from HF hub! {pretrained_model_name_or_path=}"
                ) from e

        if pretrained_model_name_or_path.endswith(".ckpt"):
            pl_ckpt_dict = None
            print(f"`.ckpt` file was located. {pretrained_model_name_or_path=}")
            if config is None:
                # Trying to get the config from the `.ckpt` parent directory
                pretrained_model_dirpath = os.path.dirname(
                    pretrained_model_name_or_path
                )
                config = os.path.join(pretrained_model_dirpath, CONFIG_NAME)

                if not os.path.exists(config):
                    print(
                        "Couldn't find `config.json` file in the checkpoint's parent directory. Trying to fetch config from the Lightning checkpoint itself"
                    )
                    pl_ckpt_dict = torch.load(
                        pretrained_model_name_or_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                    config = pl_ckpt_dict["config"]
                    for deprecated_arg in MammalConfig.get_deprecated_arguments():
                        # Remove deprecated arg if exists
                        if hasattr(config, deprecated_arg):
                            print(
                                f"Found deprecated argument '{deprecated_arg}'. Deleting it!"
                            )
                            delattr(config, deprecated_arg)

            if isinstance(config, str):
                # Config path was given or was located in the parent directory
                config = MammalConfig.from_path(
                    config_filepath=config, allow_config_mismatch=allow_config_mismatch
                )

            # override configuration if requested
            if config_overrides is not None:
                config.override(config_overrides)
            model = cls(config)

            if hasattr(config, "random_weights") and config.random_weights:
                print(
                    "Warning! You are loading random weights! To disable it, make sure to config 'random_weights' to False."
                )
            else:
                if pl_ckpt_dict is None:
                    # Didn't load it before to fetch the config. Load it now to get the weights
                    pl_ckpt_dict = torch.load(
                        pretrained_model_name_or_path,
                        map_location="cpu",
                        weights_only=True,
                    )

                state_dict = pl_ckpt_dict["state_dict"]

                lightning_model_prefix = "_model."
                state_dict = {
                    (
                        key[len(lightning_model_prefix) :]
                        if key.startswith(lightning_model_prefix)
                        else key
                    ): value
                    for key, value in state_dict.items()
                }

                pretrained_has_lora_weights = has_lora_weights(state_dict=state_dict)
                if pretrained_has_lora_weights and not config.use_lora:
                    raise ValueError(
                        "LoRA weights found in state_dict, but 'config.use_lora' is False."
                    )

                if config.use_lora and pretrained_has_lora_weights:
                    print("Loading model with existing LoRA weights in it")
                    model.t5_model = get_lora_model(model.t5_model)
                    model.load_state_dict(state_dict, strict=strict)

                elif config.use_lora and not pretrained_has_lora_weights:
                    print("Loading model and create new LoRA weights on it")
                    model.load_state_dict(state_dict, strict=strict)
                    model.t5_model = get_lora_model(model.t5_model)

                else:
                    model.load_state_dict(state_dict, strict=strict)

        elif os.path.isdir(pretrained_model_name_or_path):
            print(
                f"Attempting to load model from dir: {pretrained_model_name_or_path=}"
            )
            if config is None:
                config = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
            if isinstance(config, str):
                config = MammalConfig.from_path(
                    config_filepath=config, allow_config_mismatch=allow_config_mismatch
                )

            # override configuration if requested
            if config_overrides is not None:
                config.override(config_overrides)

            model = cls(config)

            model_safetensors_path = os.path.join(
                pretrained_model_name_or_path, SAFETENSORS_SINGLE_FILE
            )

            state_dict = load_file(model_safetensors_path)
            pretrained_has_lora_weights = has_lora_weights(state_dict=state_dict)
            if pretrained_has_lora_weights and not config.use_lora:
                raise ValueError(
                    "LoRA weights found in state_dict, but 'config.use_lora' is False."
                )

            if config.use_lora and pretrained_has_lora_weights:
                print("Loading model with existing LoRA weights in it")
                model.t5_model = get_lora_model(model.t5_model)

            if hasattr(config, "random_weights") and config.random_weights:
                print(
                    "Warning! You are using random weights! To disable it, make sure to config 'random_weights' to False."
                )
            else:
                # Inject weights to model instance
                load_model(model, model_safetensors_path, strict=strict)

            if config.use_lora and not pretrained_has_lora_weights:
                print("Loading model and create new LoRA weights on it")
                model.t5_model = get_lora_model(model.t5_model)

        else:
            raise ValueError(f"Got an invalid input, {pretrained_model_name_or_path=}")
        return model

    @property
    def device(self) -> torch.device:
        return self.t5_model.device


def get_encoder_mlp_head(
    embedding_size: int, layers: list[int], dropout: float, num_classes: int
) -> torch.nn.Module:
    return ClassifierMLP(
        in_ch=embedding_size,
        layers_description=layers,
        dropout_rate=dropout,
        num_classes=num_classes,
    )


def has_lora_weights(state_dict: dict[str, Any]) -> bool:
    """
    Checks if the given state_dict contains LoRA weights.
    """
    return any("lora" in w for w in state_dict.keys())

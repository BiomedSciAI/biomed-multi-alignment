from functools import partial
from typing import Any

import pytorch_lightning as pl
import torch
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

from mammal.examples.dti_bindingdb_kd.pl_data_module import (
    DtiBindingdbKdDataModule,
)
from mammal.keys import *
from mammal.metrics import regression_metrics
from mammal.task import (
    MammalTask,
    MetricBase,
)


class DtiBindingdbKdTask(MammalTask):
    def __init__(
        self,
        *,
        name: str,
        tokenizer_op: ModularTokenizerOp,
        data_module_kwargs: dict,
        seed: int,
        logger: Any | None = None,
        norm_y_mean: float = 0.0,
        norm_y_std: float = 1.0,
    ) -> None:
        """
        :param name: task name. used for to log metrics and losses
        :param tokenizer op: the tokenizer used
        :param data_module_kwargs: arguments for data module constructor
        :param seed: seed for random operations.
        :param logger: typically clearml logger. Optional.
        :param norm_y_mean: Used to normalize the values. Metrics will still be calculated with the original values for a fair evaluation.
                            Default value means - no normalization
        :param norm_y_std:  Used to normalize the values. Metrics will still be calculated with the original values for a fair evaluation.
                            Default value means - no normalization

        """
        super().__init__(
            name=name,
            logger=logger,
            tokenizer_op=tokenizer_op,
        )
        self._data_module_kwargs = data_module_kwargs
        self._seed = seed

        self.preds_key = CLS_PRED
        self.scores_key = SCORES
        self.labels_key = LABELS_TOKENS

        self.norm_y_mean = norm_y_mean
        self.norm_y_std = norm_y_std

    def data_module(self) -> pl.LightningDataModule:
        return DtiBindingdbKdDataModule(
            tokenizer_op=self._tokenizer_op,
            seed=self._seed,
            data_preprocessing=partial(
                self.data_preprocessing,
                norm_y_mean=self.norm_y_mean,
                norm_y_std=self.norm_y_std,
            ),
            **self._data_module_kwargs,
        )

    def train_metrics(self) -> dict[str, MetricBase]:
        metrics = super().train_metrics()
        metrics.update(
            regression_metrics(
                self.name(),
                process_func=partial(
                    self.process_model_output,
                    norm_y_mean=self.norm_y_mean,
                    norm_y_std=self.norm_y_std,
                ),
                pred_scalars_key="model.out.dti_bindingdb_kd",
                target_scalars_key="Y",
            )
        )

        return metrics

    def validation_metrics(self) -> dict[str, MetricBase]:
        validation_metrics = super().validation_metrics()
        validation_metrics.update(
            regression_metrics(
                self.name(),
                process_func=partial(
                    self.process_model_output,
                    norm_y_mean=self.norm_y_mean,
                    norm_y_std=self.norm_y_std,
                ),
                pred_scalars_key="model.out.dti_bindingdb_kd",
                target_scalars_key="Y",
            )
        )
        return validation_metrics

    @staticmethod
    def data_preprocessing(
        sample_dict: dict,
        *,
        target_sequence_key: str,
        drug_sequence_key: str,
        ground_truth_key: int | None = None,
        target_max_seq_length: int = 1250,
        drug_max_seq_length: int = 256,
        encoder_input_max_seq_len: int = 1512,
        tokenizer_op: ModularTokenizerOp,
        norm_y_mean: float,
        norm_y_std: float,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """
        :param norm_y_mean: Used to normalize the values. Metrics will still be calculated with the original values for a fair evaluation.
                            Default value means - no normalization
        :param norm_y_std:  Used to normalize the values. Metrics will still be calculated with the original values for a fair evaluation.
                            Default value means - no normalization
        """
        target_sequence = sample_dict[target_sequence_key]
        drug_sequence = sample_dict[drug_sequence_key]
        ground_truth_value = sample_dict.get(ground_truth_key, None)

        sample_dict[ENCODER_INPUTS_STR] = (
            "<@TOKENIZER-TYPE=AA><MASK>"
            f"<@TOKENIZER-TYPE=AA@MAX-LEN={target_max_seq_length}><MOLECULAR_ENTITY><MOLECULAR_ENTITY_GENERAL_PROTEIN><SEQUENCE_NATURAL_START>{target_sequence}<SEQUENCE_NATURAL_END>"
            f"<@TOKENIZER-TYPE=SMILES@MAX-LEN={drug_max_seq_length}><MOLECULAR_ENTITY><MOLECULAR_ENTITY_SMALL_MOLECULE><SEQUENCE_NATURAL_START>{drug_sequence}<SEQUENCE_NATURAL_END>"
            "<EOS>"
        )
        tokenizer_op(
            sample_dict,
            key_in=ENCODER_INPUTS_STR,
            key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
            key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK,
            max_seq_len=encoder_input_max_seq_len,
            key_out_scalars=ENCODER_INPUTS_SCALARS,
        )

        sample_dict[ENCODER_INPUTS_TOKENS] = torch.tensor(
            sample_dict[ENCODER_INPUTS_TOKENS], device=device
        )
        sample_dict[ENCODER_INPUTS_ATTENTION_MASK] = torch.tensor(
            sample_dict[ENCODER_INPUTS_ATTENTION_MASK],
            device=device,
        )

        if ground_truth_value is not None:
            ground_truth_value = (ground_truth_value - norm_y_mean) / norm_y_std
            pad_id = tokenizer_op.get_token_id("<PAD>")
            ignore_token_value = -100
            sample_dict[LABELS_STR] = (
                f"<@TOKENIZER-TYPE=SCALARS_LITERALS>{ground_truth_value}<@TOKENIZER-TYPE=AA>"
                + "".join(["<PAD>"] * (encoder_input_max_seq_len - 1))
            )

            tokenizer_op(
                sample_dict,
                key_in=LABELS_STR,
                key_out_tokens_ids=LABELS_TOKENS,
                key_out_attention_mask=LABELS_ATTENTION_MASK,
                max_seq_len=encoder_input_max_seq_len,
                key_out_scalars=LABELS_SCALARS,
                validate_ends_with_eos=False,
            )

            sample_dict[LABELS_TOKENS] = torch.tensor(
                sample_dict[LABELS_TOKENS], device=device
            )
            sample_dict[LABELS_ATTENTION_MASK] = torch.tensor(
                sample_dict[LABELS_ATTENTION_MASK], device=device
            )
            # replace pad_id with -100 to
            pad_id_tns = torch.tensor(pad_id)
            sample_dict[LABELS_TOKENS][
                (sample_dict[LABELS_TOKENS][..., None] == pad_id_tns).any(-1).nonzero()
            ] = ignore_token_value

            sample_dict[LABELS_SCALARS_VALUES] = sample_dict[LABELS_SCALARS_VALUES].to(
                device=device
            )
            sample_dict[LABELS_SCALARS_VALID_MASK] = sample_dict[
                LABELS_SCALARS_VALID_MASK
            ].to(device=device)

        return sample_dict

    @staticmethod
    def process_model_output(
        batch_dict: dict,
        *,
        scalars_preds_key: str = SCALARS_PREDICTION_HEAD_LOGITS,
        scalars_preds_processed_key: str = "model.out.dti_bindingdb_kd",
        norm_y_mean: float,
        norm_y_std: float,
    ) -> dict:
        """
        :param norm_y_mean: Used to normalize the values. Metrics will still be calculated with the original values for a fair evaluation.
                            Default value means - no normalization
        :param norm_y_std:  Used to normalize the values. Metrics will still be calculated with the original values for a fair evaluation.
                            Default value means - no normalization
        """
        scalars_preds = batch_dict[scalars_preds_key]

        batch_dict[scalars_preds_processed_key] = (
            scalars_preds[:, 0] * norm_y_std + norm_y_mean
        )

        return batch_dict

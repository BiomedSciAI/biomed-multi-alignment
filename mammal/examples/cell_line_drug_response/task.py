"""
Task definition for cell line drug response (IC50) prediction fine-tuning.

This task uses MAMMAL's encoder-only mode with scalar prediction head for regression.
"""

from typing import Any

import pytorch_lightning as pl
import torch
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

from mammal.examples.cell_line_drug_response.dataset import (
    sort_genes_by_value_and_name,
)
from mammal.examples.cell_line_drug_response.pl_data_module import (
    CellLineDrugResponseDataModule,
)
from mammal.keys import *
from mammal.metrics import regression_metrics
from mammal.task import MammalTask, MetricBase


class CellLineDrugResponseTask(MammalTask):
    """
    Task for predicting IC50 values for drug-cell line combinations using GDSC dataset.

    Uses encoder-only mode with auxiliary scalar prediction head for regression.
    Metrics include Pearson correlation, Spearman correlation, MSE, and R².
    """

    def __init__(
        self,
        *,
        name: str,
        tokenizer_op: ModularTokenizerOp,
        data_module_kwargs: dict,
        seed: int,
        logger: Any | None = None,
    ) -> None:
        super().__init__(
            name=name,
            logger=logger,
            tokenizer_op=tokenizer_op,
        )
        self._data_module_kwargs = data_module_kwargs
        self._seed = seed

    def data_module(self) -> pl.LightningDataModule:
        """Create the PyTorch Lightning data module."""
        return CellLineDrugResponseDataModule(
            tokenizer_op=self._tokenizer_op,
            seed=self._seed,
            data_preprocessing=self.data_preprocessing,
            **self._data_module_kwargs,
        )

    def train_metrics(self) -> dict[str, MetricBase]:
        """Define training metrics."""
        metrics = super().train_metrics()
        metrics.update(
            regression_metrics(
                self.name(),
                process_func=self.process_model_output,
                pred_scalars_key="model.out.ic50_pred",
                target_scalars_key="Y",
            )
        )
        return metrics

    def validation_metrics(self) -> dict[str, MetricBase]:
        """Define validation metrics."""
        validation_metrics = super().validation_metrics()
        validation_metrics.update(
            regression_metrics(
                self.name(),
                process_func=self.process_model_output,
                pred_scalars_key="model.out.ic50_pred",
                target_scalars_key="Y",
            )
        )
        return validation_metrics

    @staticmethod
    def data_preprocessing(
        sample_dict: dict,
        *,
        genes_key: str,
        expressions_key: str,
        drug_smiles_key: str,
        ground_truth_key: str | None = None,
        tokenizer_op: ModularTokenizerOp,
        encoder_input_max_seq_len: int = 1560,
        max_genes: int = 1295,
        device: str | torch.device = "cpu",
    ) -> dict:
        genes = sample_dict[genes_key]
        expressions = sample_dict[expressions_key]
        drug_smiles = sample_dict[drug_smiles_key]
        ground_truth_value = sample_dict.get(ground_truth_key, None)

        genes_sorted = sort_genes_by_value_and_name(expressions, genes)[:max_genes]
        genes_formatted = [f"[{gene}]" for gene in genes_sorted]

        sample_dict[ENCODER_INPUTS_STR] = (
            "<@TOKENIZER-TYPE=SMILES><MASK>"
            f"<@TOKENIZER-TYPE=SMILES><MOLECULAR_ENTITY><MOLECULAR_ENTITY_SMALL_MOLECULE><SMILES_SEQUENCE>{drug_smiles}"
            "<@TOKENIZER-TYPE=GENE><MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>"
            + "".join(genes_formatted)
            + "<EOS>"
        )

        tokenizer_op(
            sample_dict=sample_dict,
            key_in=ENCODER_INPUTS_STR,
            key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
            key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK,
            key_out_scalars=ENCODER_INPUTS_SCALARS,
            max_seq_len=encoder_input_max_seq_len,
            on_unknown="warn",
            verbose=0,
        )

        sample_dict[ENCODER_INPUTS_TOKENS] = torch.tensor(
            sample_dict[ENCODER_INPUTS_TOKENS], dtype=torch.long, device=device
        )
        sample_dict[ENCODER_INPUTS_ATTENTION_MASK] = torch.tensor(
            sample_dict[ENCODER_INPUTS_ATTENTION_MASK], dtype=torch.long, device=device
        )

        if ground_truth_value is not None:
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
        scalars_preds_processed_key: str = "model.out.ic50_pred",
    ) -> dict:
        """
        Process model output to extract IC50 predictions.

        Args:
            batch_dict: Batch dictionary with model outputs
            scalars_preds_key: Key for raw scalar predictions
            scalars_preds_processed_key: Key to store processed predictions

        Returns:
            Updated batch dictionary with processed predictions
        """
        scalars_preds = batch_dict[scalars_preds_key]
        batch_dict[scalars_preds_processed_key] = scalars_preds[:, 0]
        return batch_dict

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

from mammal.examples.scrna_cell_type.pl_data_module import (
    CellTypeDataModule,
)
from mammal.keys import (
    CLS_PRED,
    DECODER_INPUTS_ATTENTION_MASK,
    DECODER_INPUTS_STR,
    DECODER_INPUTS_TOKENS,
    ENCODER_INPUTS_ATTENTION_MASK,
    ENCODER_INPUTS_STR,
    ENCODER_INPUTS_TOKENS,
    LABELS_ATTENTION_MASK,
    LABELS_STR,
    LABELS_TOKENS,
    SCORES,
)
from mammal.metrics import classification_metrics
from mammal.task import (
    MammalTask,
    MetricBase,
)

ALL_CLASS_LABELS = [
    "[CL:0000794]",
    "[CL:0001062]",
    "[CL:0000939]",
    "[CL:0000792]",
    "[CL:0000236]",
    "[CL:0001204]",
    "[CL:0001054]",
    "[CL:0000451]",
    "[CL:0000895]",
    "[CL:0000049]",
    "[CL:0000546]",
]


class CellTypeTask(MammalTask):
    def __init__(
        self,
        *,
        tokenizer_op: ModularTokenizerOp,
        data_module_kwargs: dict,
        logger: Any | None = None,
    ) -> None:
        super().__init__(
            name="cell_type",
            logger=logger,
            tokenizer_op=tokenizer_op,
        )
        self._data_module_kwargs = data_module_kwargs

        self.preds_key = CLS_PRED
        self.scores_key = SCORES
        self.labels_key = LABELS_TOKENS

    def data_module(self) -> pl.LightningDataModule:
        return CellTypeDataModule(
            tokenizer_op=self._tokenizer_op,
            data_preprocessing=self.data_preprocessing,
            stratify_by=["label"],
            **self._data_module_kwargs,
        )

    def train_metrics(self) -> dict[str, MetricBase]:
        metrics = super().train_metrics()
        metrics.update(
            # TODO: update this
            classification_metrics(
                self.name(),
                class_position=1,
                tokenizer_op=self._tokenizer_op,
                class_tokens=ALL_CLASS_LABELS,
            )
        )

        return metrics

    def validation_metrics(self) -> dict[str, MetricBase]:
        validation_metrics = super().validation_metrics()
        validation_metrics.update(
            classification_metrics(
                self.name(),
                class_position=1,
                tokenizer_op=self._tokenizer_op,
                class_tokens=ALL_CLASS_LABELS,
            )
        )
        return validation_metrics

    @staticmethod
    def data_preprocessing(
        sample_dict: dict,
        *,
        sequence_key: str,
        label_key: int | None = None,
        # drug_max_seq_length: int = 1250,
        input_max_seq_length: int | None = 1260,
        encoder_input_max_seq_len: int | None = 1260,
        labels_max_seq_len: int | None = 4,
        tokenizer_op: ModularTokenizerOp,
    ) -> dict:
        """process a sample into the format expected by the model

        Args:
            sample_dict (dict): dictonary with the sample data
            sequence_key (str): key in the dictionary with the sequence
            tokenizer_op (ModularTokenizerOp): the tokenizer
            label_key (int | None, optional): key for the lable. Defaults to None.
            input_max_seq_length (int | None, optional): sequance is trancated if longer than this. Defaults to 1260.
            encoder_input_max_seq_len (int | None, optional): maximal length of encoder input. Defaults to 1260.
            labels_max_seq_len (int | None, optional): maximal length of lable sequance. Defaults to 4.

        Returns:
            dict: the sample dict with added keys and values:

        Mammal model expects a dictionary with a set of keys to be able to run.  This method converts the data into the expected format.
        Here is a list of the required fields for an encoder-decoder task:
            ENCODER_INPUTS_STR
            ENCODER_INPUTS_TOKENS
            ENCODER_INPUTS_ATTENTION_MASK

            LABELS_STR
            LABELS_TOKENS
            LABELS_ATTENTION_MASK

            DECODER_INPUTS_STR
            DECODER_INPUTS_TOKENS
            DECODER_INPUTS_ATTENTION_MASK

            see MammalTask.data_module for more information about these keys and their use.


        The three *_str values are constricted here, and then the others are derived from them by the tokenizer_op
        """
        scrna = sample_dict[sequence_key]
        if label_key:
            label = sample_dict.get(label_key, None)
        else:
            label = None

        # we have a link to the data of the specific cell, as a refrence into the anndata objerct
        # To get the canonical gene names we need to get access to the anndata object itself.
        gene_names = scrna._view_args.parent.var_names.to_numpy()

        # This is where the data is converted to GeneFormer inspired "binned and sorted"
        # The binning is done in preprocess_ann_data, on load rather then when training.
        # The sorting is done first over the binned expression values and then on the gene names
        # This is achived by zipping together the minus the bin (so to sort it from large to small)
        # and the standertized gene name.
        # scrna.data are the non-zero values of the raw, scrna.indices are the indexes for these values

        sorted_genes = [
            a[1] for a in sorted(zip(-scrna.data, gene_names[scrna.indices]))
        ]
        sequence_string = "[" + "][".join(sorted_genes[:input_max_seq_length]) + "]"

        sample_dict[ENCODER_INPUTS_STR] = (
            f"<@TOKENIZER-TYPE=GENE><MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED><{sequence_string}<EOS>"
        )

        tokenizer_op(
            sample_dict=sample_dict,
            key_in=ENCODER_INPUTS_STR,
            key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
            key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK,
            max_seq_len=encoder_input_max_seq_len,
        )
        sample_dict[ENCODER_INPUTS_TOKENS] = torch.tensor(
            sample_dict[ENCODER_INPUTS_TOKENS]
        )
        sample_dict[ENCODER_INPUTS_ATTENTION_MASK] = torch.tensor(
            sample_dict[ENCODER_INPUTS_ATTENTION_MASK]
        )

        if label is not None:
            pad_id = tokenizer_op.get_token_id("<PAD>")
            ignore_token_value = -100
            sample_dict[LABELS_STR] = (
                f"<@TOKENIZER-TYPE=CELL_ATTRIBUTES><SENTINEL_ID_0>[{label}]<EOS>"
            )
            tokenizer_op(
                sample_dict=sample_dict,
                key_in=LABELS_STR,
                key_out_tokens_ids=LABELS_TOKENS,
                key_out_attention_mask=LABELS_ATTENTION_MASK,
                max_seq_len=labels_max_seq_len,
            )
            sample_dict[LABELS_TOKENS] = torch.tensor(sample_dict[LABELS_TOKENS])
            sample_dict[LABELS_ATTENTION_MASK] = torch.tensor(
                sample_dict[LABELS_ATTENTION_MASK]
            )
            # replace pad_id with -100 to
            sample_dict[LABELS_TOKENS][
                (sample_dict[LABELS_TOKENS][..., None] == torch.tensor(pad_id))
                .any(-1)
                .nonzero()
            ] = ignore_token_value

            sample_dict[DECODER_INPUTS_STR] = (
                f"<@TOKENIZER-TYPE=CELL_ATTRIBUTES><DECODER_START><SENTINEL_ID_0><{label}><EOS>"
            )
            tokenizer_op(
                sample_dict=sample_dict,
                key_in=DECODER_INPUTS_STR,
                key_out_tokens_ids=DECODER_INPUTS_TOKENS,
                key_out_attention_mask=DECODER_INPUTS_ATTENTION_MASK,
                max_seq_len=labels_max_seq_len,
            )
            sample_dict[DECODER_INPUTS_TOKENS] = torch.tensor(
                sample_dict[DECODER_INPUTS_TOKENS]
            )
            sample_dict[DECODER_INPUTS_ATTENTION_MASK] = torch.tensor(
                sample_dict[DECODER_INPUTS_ATTENTION_MASK]
            )

        return sample_dict

    @staticmethod
    def process_model_output(
        tokenizer_op: ModularTokenizerOp,
        decoder_output: np.ndarray,
        decoder_output_scores: np.ndarray,
    ) -> dict | None:
        ans = None
        all_class_label_ids = [
            tokenizer_op.get_token_id(class_label) for class_label in ALL_CLASS_LABELS
        ]
        classification_position = 1
        if decoder_output_scores is not None:
            class_scores = decoder_output_scores[classification_position][
                all_class_label_ids
            ]
            best_match = class_scores.argmax()
            non_normelized_score = class_scores[best_match]
            normalization_factor = class_scores.sum()
            normalized_score = non_normelized_score / (
                normalization_factor + 1e-30
            )  # incase non seem to match
            ans = {
                "cell_type": ALL_CLASS_LABELS[best_match],
                "pred": all_class_label_ids[best_match],
                "not_normalized_scores": non_normelized_score,
                "normalized_scores": normalized_score,
            }

        return ans

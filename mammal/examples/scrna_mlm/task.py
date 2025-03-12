from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

from mammal.examples.scrna_cell_type.pl_data_module import (
    CellTypeDataModule,
)
from mammal.examples.scrna_mlm.anndata_op import OpMaskedSeqToMLM, OpRandomMaskVector
from mammal.keys import (
    CLS_PRED,
    ENCODER_INPUTS_ATTENTION_MASK,
    ENCODER_INPUTS_STR,
    ENCODER_INPUTS_TOKENS,
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


class ScRNAMLMTask(MammalTask):
    def __init__(
        self,
        *,
        tokenizer_op: ModularTokenizerOp,
        data_module_kwargs: dict,
        logger: Any | None = None,
    ) -> None:
        super().__init__(
            name="scrna_mlm",
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
        input_max_seq_length: int | None = 1260,
        encoder_input_max_seq_len: int | None = 1260,
        mask_prob: float = 0.1,
        seed: float = 42,
        tokenizer_op: ModularTokenizerOp,
    ):
        """process a sample into the format expected by the model

        Args:
            sample_dict (dict): dictionary with the sample data
            sequence_key (str): key in the dictionary with the sequence
            tokenizer_op (ModularTokenizerOp): the tokenizer
            input_max_seq_length (int | None, optional): sequence is truncated if longer than this. Defaults to 1260.
            encoder_input_max_seq_len (int | None, optional): maximal length of encoder input. Defaults to 1260.
            labels_max_seq_len (int | None, optional): maximal length of label sequence. Defaults to 4.

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

        # we have a link to the data of the specific cell, as a reference into the AnnData object
        # To get the canonical gene names we need to get access to the AnnData object itself.
        gene_names = scrna._view_args.parent.var_names.to_numpy()

        # This is where the data is converted to GeneFormer inspired "binned and sorted"
        # The binning is done in preprocess_ann_data, on load rather then when training.

        sorted_genes = ScRNAMLMTask.convert_to_double_sorted_geneformer_sequence(
            scrna_sample=scrna, gene_names=gene_names
        )
        sequence_string = "[" + "][".join(sorted_genes[:input_max_seq_length]) + "]"

        sample_dict[ENCODER_INPUTS_STR] = (
            f"<@TOKENIZER-TYPE=GENE><MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED><{sequence_string}<EOS>"
        )
        sample_dict["sorted_genes"] = [
            f"[{v}]" for v in sorted_genes[:input_max_seq_length]
        ]
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
        mask_op = OpRandomMaskVector(
            mask_prob=mask_prob,
            random_seed=seed,
            key_in_vec="sorted_genes",  # "data.query.encoder_input",
            key_out_masked_seq="data.masked_seq",
        )
        mask_op(sample_dict)
        create_op = OpMaskedSeqToMLM(tokenizer_type="GENE")
        create_op(
            sample_dict, key_in_masked_seq="data.masked_seq", key_out="data.mlm_format"
        )

        return sample_dict

    @staticmethod
    def convert_to_double_sorted_geneformer_sequence(scrna_sample, gene_names):
        """convert binned genes to double sorted GeneFormer like format.
        The sorting is done first over the binned expression values and then on the gene names
        This is achieved by zipping together the minus the bin (so to sort it from large to small)
        and the standardized gene name.
        sample.data are the non-zero values of the raw, sample.indices are the indexes for these values


        Args:
            sample: Dataframe with gene bins and matching indexes
            gene_names (list[str]):list of gene names matching the list above

        Returns:
            list[str] - gene names sorted by bin values and then by gene name
        """
        return [
            a[1]
            for a in sorted(zip(-scrna_sample.data, gene_names[scrna_sample.indices]))
        ]

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
            non_normalized_score = class_scores[best_match]
            normalization_factor = class_scores.sum()
            normalized_score = non_normalized_score / (
                normalization_factor + 1e-30
            )  # incase non seem to match
            ans = {
                "cell_type": ALL_CLASS_LABELS[best_match],
                "pred": all_class_label_ids[best_match],
                "not_normalized_scores": non_normalized_score,
                "normalized_scores": normalized_score,
            }

        return ans

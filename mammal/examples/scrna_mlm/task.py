from typing import Any
import torch
import pytorch_lightning as pl
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

from mammal.examples.scrna_mlm.anndata_op import OpMaskedSeqToMLM, OpRandomMaskVector
from mammal.examples.scrna_mlm.pl_data_module import ScRNAMLMDataModule
from mammal.keys import *
from mammal.task import (
    MammalTask,
    MetricBase,
)


class ScRNAMLMTask(MammalTask):
    def __init__(
        self,
        *,
        tokenizer_op: ModularTokenizerOp,
        data_module_kwargs: dict,
        logger: Any | None = None,
        mask_prob: float = 0.1,
        seed: int = 42,
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
        self.mask_op = OpRandomMaskVector(
            mask_prob=mask_prob,
            random_seed=seed,
            key_in_vec="sorted_genes",  # "data.query.encoder_input",
            key_out_masked_seq="data.masked_seq",
        )
        self.create_op = OpMaskedSeqToMLM(tokenizer_type="GENE")

    def data_module(self) -> pl.LightningDataModule:
        return ScRNAMLMDataModule(
            tokenizer_op=self._tokenizer_op,
            data_preprocessing=self.data_preprocessing,
            **self._data_module_kwargs,
        )

    def train_metrics(self) -> dict[str, MetricBase]:
        metrics = super().train_metrics()
        return metrics

    def validation_metrics(self) -> dict[str, MetricBase]:
        validation_metrics = super().validation_metrics()
        return validation_metrics

    def data_preprocessing(
        self,
        sample_dict: dict,
        *,
        sequence_key: str,
        input_max_seq_length: int | None = 1260,
        encoder_input_max_seq_len: int | None = 1260,
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
        sample_dict["sorted_genes"] = [
            f"[{v}]" for v in sorted_genes[:input_max_seq_length]
        ]
        self.mask_op(sample_dict)
        self.create_op(
            sample_dict, key_in_masked_seq="data.masked_seq", key_out="data.query"
        )
        tokenizer_op(
            sample_dict=sample_dict,
            key_in=ENCODER_INPUTS_STR,
            key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
            key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK,
            max_seq_len=encoder_input_max_seq_len,
        )
        tokenizer_op(
            sample_dict=sample_dict,
            key_in=DECODER_INPUTS_STR,
            key_out_tokens_ids=DECODER_INPUTS_TOKENS,
            key_out_attention_mask=DECODER_INPUTS_ATTENTION_MASK,
            max_seq_len=encoder_input_max_seq_len,
        )
        tokenizer_op(
            sample_dict=sample_dict,
            key_in=LABELS_STR,
            key_out_tokens_ids=LABELS_TOKENS,
            key_out_attention_mask=LABELS_ATTENTION_MASK,
            max_seq_len=encoder_input_max_seq_len,
        )
        for entry in [DECODER_INPUTS_TOKENS,DECODER_INPUTS_ATTENTION_MASK,ENCODER_INPUTS_TOKENS,ENCODER_INPUTS_ATTENTION_MASK,LABELS_TOKENS,LABELS_ATTENTION_MASK]:
            sample_dict[entry] = torch.tensor(
                sample_dict[entry]
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

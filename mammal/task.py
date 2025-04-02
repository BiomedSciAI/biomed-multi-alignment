from collections import OrderedDict
from typing import Any

import pytorch_lightning as pl
import torch
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from fuse.eval import MetricBase
from fuse.eval.metrics.sequence_gen.metrics_seq_gen_common import MetricPerplexity

from mammal.keys import *  # noqa
from mammal.losses import LossHead, ScalarsPredictionsLoss
from mammal.metrics import MetricSeqAccuracy


class MammalTask:
    """
    A class that holds all the requirements to define a new task.
    A new task expected to inherit and override all the necessary methods.
    """

    def __init__(
        self,
        *,
        name: str,
        logger: Any,
        tokenizer_op: ModularTokenizerOp,
        scalars_loss_weight: float = 1.0,
        scalars_loss_type: str = "mse",
    ) -> None:
        """
        Args:
        :param loss: controls which loss function is used. Supported options are: ['ce', 'focal']
            Note: for full control over the calculated loss(es), you can override the method "losses"
        :param loss_weight: multiplication factor for the loss.
        """
        self._logger = logger
        self._tokenizer_op = tokenizer_op
        self._name = name
        self._scalars_loss_weight = scalars_loss_weight
        self._scalars_loss_type = scalars_loss_type

    def name(self) -> str:
        return self._name

    def data_module(self) -> pl.LightningDataModule:
        """
        Return a lightning data module for the task.
        The dataloaders implemented in this datamodule expect to iterate over batches.
        Each batch represented by a dict.
        In the dictionary the following key-value pairs must be set:

        mammal.keys.ENCODER_INPUTS_STR # the original string representation of encoder input - used for debug
        mammal.keys.ENCODER_INPUTS_TOKENS # encoder input token ids
        mammal.keys.ENCODER_INPUTS_ATTENTION_MASK  # attention mask of the tokenized encoder input (output of the tokenizer)

        mammal.keys.LABELS_STR # the original string representation of labels - used for debug
        mammal.keys.LABELS_TOKENS  # labels token ids
        mammal.keys.LABELS_ATTENTION_MASK  # attention mask of the tokenized labels (output of the tokenizer)

        In an encoder-decoder mode, also the following expected to be set:
        mammal.keys.DECODER_INPUTS_STR # the original string representation of decoder input - used for debug
        mammal.keys.DECODER_INPUTS_TOKENS  # decoder input token ids (decoder start token followed by labels token ids)
        mammal.keys.DECODER_INPUTS_ATTENTION_MASK  # attention mask of the tokenized decoder input (output of the tokenizer)

        """
        raise NotImplementedError()

    def losses(self) -> dict[str, torch.nn.Module]:
        """
        Returns dictionary of losses. The total loss will be the sum of all losses.
        Each loss element represented by a pytorch module that gets a batch represented by a dictionary
        The implementation is typical and work for most cases.
        It is the sum of a cross-entropy loss applied on any label != -100 and an mse of any scalar value.
        """
        all_losses = {}

        loss_object = LossHead(loss_type="ce")
        all_losses[f"{self.name()}_ce"] = loss_object

        # scalars
        loss_object = ScalarsPredictionsLoss(
            loss_type=self._scalars_loss_type,
            loss_weight=self._scalars_loss_weight,
        )
        all_losses[f"{self.name()}_scalars_mse"] = loss_object

        return all_losses

    def train_metrics(self) -> dict[str, MetricBase]:
        """
        Fuse Metrics for trainset
        """
        return self.get_metrics(is_train=True)

    def validation_metrics(self) -> dict[str, MetricBase]:
        """
        Fuse Metrics for validationset
        """
        return self.get_metrics(is_train=False)

    def get_metrics(self, is_train: bool) -> dict[str, MetricBase]:
        """
        Default metrics to use: Perplexity, and Token Accuracy
        """
        return OrderedDict(
            [
                (
                    f"{self.name()}_perplexity",
                    MetricPerplexity(
                        preds=SCORES, target=LABELS_TOKENS, ignore_index=-100
                    ),
                ),
                (
                    f"{self.name()}_token_acc",
                    MetricSeqAccuracy(pred=CLS_PRED, target=LABELS_TOKENS),
                ),
            ]
        )

    @staticmethod
    def data_preprocessing(sample_dict: dict, *args: list, **kwargs: dict) -> str:
        """
        The point of this method is to get a task specific input (in a way that is easy to provide, for example, AA sequences),
         and to construct a query for the model.
        A query built from encoder_input and for if available for training also labels and decoder_input
        See examples in mammal/examples/protein_solubility/task.py

        This function also responsible to tokenize the query and to set any label that should participate in loss to -100.

        The function will get sample_dict with all the raw sample data and should add the following keys.

        mammal.keys.ENCODER_INPUTS_STR # the original string representation of encoder input - used for debug
        mammal.keys.ENCODER_INPUTS_TOKENS # encoder input token ids
        mammal.keys.ENCODER_INPUTS_ATTENTION_MASK  # attention mask of the tokenized encoder input (output of the tokenizer)

        And if available for training also:
        mammal.keys.DECODER_INPUTS_STR # the original string representation of decoder input - used for debug
        mammal.keys.DECODER_INPUTS_TOKENS  # decoder input token ids (decoder start token followed by labels token ids)
        mammal.keys.DECODER_INPUTS_ATTENTION_MASK  # attention mask of the tokenized decoder input (output of the tokenizer)

        mammal.keys.LABELS_STR # the original string representation of labels - used for debug
        mammal.keys.LABELS_TOKENS  # labels token ids
        mammal.keys.LABELS_ATTENTION_MASK  # attention mask of the tokenized labels (output of the tokenizer)

        """
        raise NotImplementedError()

    @staticmethod
    def process_model_output(
        tokenizer_op: ModularTokenizerOp, verbose: bool = False, **kwargs: dict
    ) -> dict:
        """
        The point of this method is to process model output in a way that extract the key meaningful values from it.
        Some task will not expect encoder_output (encoder-decoder tasks) and some task will not expect decoder_output (encoder-only tasks)
        logits is expected in tasks that have predictive aspects, for example in binding binary classification.

        See examples in mammal/examples/protein_solubility/task.py

        """
        raise NotImplementedError()

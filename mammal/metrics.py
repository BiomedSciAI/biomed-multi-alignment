from collections.abc import Callable
from functools import partial

import numpy as np
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from fuse.eval.metrics.classification.metrics_classification_common import (
    MetricAccuracy,
    MetricAUCROC,
    MetricMCC,
)
from fuse.eval.metrics.metrics_common import MetricBase, MetricDefault
from fuse.eval.metrics.regression.metrics import (
    MetricMAE,
    MetricMSE,
    MetricPearsonCorrelation,
    MetricR2,
    MetricRMSE,
    MetricSpearmanCorrelation,
)

from mammal.keys import *

"""
Generic LM metrics used by default for each task in this generic multitask t5 implementation
"""


class MetricSeqAccuracy(MetricDefault):
    """
    Accuracy of a generated sequence. (Token Accuracy)
    """

    def __init__(
        self,
        pred: str | None = None,
        target: str | None = None,
        ignore_index: int | None = -100,
        **kwargs: dict,
    ):
        """
        :param pred: batch_dict key to class! predictions
        :param target: batch_dict to labels
        :param ignore_index: ignore this label values in the returned score
        """
        super().__init__(
            metric_func=self.sequence_accuracy, pred=pred, target=target, **kwargs
        )
        self.ignore_index = ignore_index

    def sequence_accuracy(
        self,
        pred: list[np.ndarray],
        target: list[np.ndarray],
        sample_weight: list[np.ndarray] | None = None,
    ) -> float:
        if isinstance(pred, list):
            pred = np.concatenate(pred)
            target = np.concatenate(target)
            if sample_weight is not None:
                sample_weight = np.concatenate(sample_weight)

        assert pred.shape == target.shape, f"shape does not match {pred.shape=} <> {target.shape=}"  # type: ignore[attr-defined]

        indices = target != self.ignore_index
        are_same_indicators = pred[indices] == target[indices]
        if sample_weight is None:
            return np.sum(are_same_indicators) / np.sum(indices)

        sample_weight = sample_weight[indices]
        return np.sum(are_same_indicators * sample_weight) / (np.sum(sample_weight))


def classification_metrics(
    name: str,
    class_position: int,
    class_tokens: list[str],
    tokenizer_op: ModularTokenizerOp,
    scores_key: str = SCORES,
    cls_preds_key: str = CLS_PRED,
    labels_key: str = LABELS_TOKENS,
) -> dict[str, MetricBase]:
    """
    Recommended metrics for classification AUC, Accuracy and MCC
    :param name: task name
    :param class_position: position (index) in the labels/predictions sequence length of the classification token
    :param class_tokens: list of possible class tokens
    :param tokenizer_op: tokenizer instance
    :param scores_key: batch_dict key that points to predictions scores
    :param cls_preds_key: batch dict key that points to predictions
    :param labels: batch dict key that points to labels
    """
    metrics = {}

    class_token_ids = [
        tokenizer_op.get_token_id(class_token) for class_token in class_tokens
    ]
    token_id_to_class_index = {
        token_id: cls_index for cls_index, token_id in enumerate(class_token_ids)
    }

    # this mode assumes the predicted class is always on a specific label
    pre_collect_fn = partial(
        extract_classification_predictions_and_labels,
        class_token_ids=class_token_ids,
        token_id_to_class_index=token_id_to_class_index,
        seq_pos=class_position,
    )

    metrics[f"{name}_aucroc"] = MetricAUCROC(
        pred=scores_key,
        target=labels_key,
        batch_pre_collect_process_func=pre_collect_fn,
    )
    metrics[f"{name}_acc"] = MetricAccuracy(
        pred=cls_preds_key,
        target=labels_key,
        batch_pre_collect_process_func=pre_collect_fn,
    )
    metrics[f"{name}_mcc"] = MetricMCC(
        pred=cls_preds_key,
        target=labels_key,
        batch_pre_collect_process_func=pre_collect_fn,
    )
    return metrics


# extract specific positions from a batch_dict according to label/out_key
def extract_classification_predictions_and_labels(
    batch_dict: dict,
    *,
    class_token_ids: list[int],
    token_id_to_class_index: dict[int, int],
    labels_key: str = LABELS_TOKENS,
    cls_preds_key: str = CLS_PRED,
    scores_key: str = SCORES,
    seq_pos: int = 1,
) -> dict:
    """
    Extract the predictions and labels and convert them from vocabulary space to class index space
    This function currently optimized for a single gpu. For multi-gpu the returned tensors should be moved back to gpu.
    :param class_token_ids: list of ids of the class tokens
    :param token_id_to_class_index: mapping from token-id to index in class_token_ids.
    :param labels_key: batch_dict key which points to labels
    :param cls_preds_key: batch_dict key which points to cls_preds
    :param scores_key: batch_dict key which points to scores
    :param seq_pos: the position of the class token in labels
    """
    device_labels = batch_dict[labels_key].device
    device_cls_preds = batch_dict[cls_preds_key].device

    labels = batch_dict[labels_key][:, seq_pos].cpu()
    cls_preds = batch_dict[cls_preds_key][:, seq_pos].contiguous().cpu()
    scores = batch_dict[scores_key][:, seq_pos, class_token_ids].contiguous()

    classification_labels = labels.apply_(
        lambda x: token_id_to_class_index.get(x, len(class_token_ids))
    )
    classification_cls_preds = cls_preds.apply_(
        lambda x: token_id_to_class_index.get(x, len(class_token_ids))
    )

    return {
        labels_key: classification_labels.to(device=device_labels),
        cls_preds_key: classification_cls_preds.to(device=device_cls_preds),
        scores_key: scores,
    }


def regression_metrics(
    name: str,
    pred_scalars_key: str = SCALARS_PREDICTION_HEAD_LOGITS,
    target_scalars_key: str = LABELS_SCALARS_VALUES,
    process_func: Callable | None = None,
) -> dict[str, MetricBase]:
    """
    Typical metrics for regression tasks: includes MetricPearsonCorrelation, MetricSpearmanCorrelation, MetricMAE, MetricMSE, MetricRMSE, MetricR2
    :param pred_scalars_key: key to scalar prediction (after it was extracted from model output)
    :param target_scalars_key: key to ground truth scalar.
    :param process_func: a function that extract the actual relevant scalar from model output and store it in batch_dict.
    """
    metrics = {}
    metrics[f"{name}_pcorr"] = MetricPearsonCorrelation(
        pred=pred_scalars_key,
        target=target_scalars_key,
        batch_pre_collect_process_func=process_func,
        mask=None,
    )
    metrics[f"{name}_spearcorr"] = MetricSpearmanCorrelation(
        pred=pred_scalars_key,
        target=target_scalars_key,
        batch_pre_collect_process_func=process_func,
        mask=None,
    )
    metrics[f"{name}_mae"] = MetricMAE(
        pred=pred_scalars_key,
        target=target_scalars_key,
        batch_pre_collect_process_func=process_func,
    )
    metrics[f"{name}_mse"] = MetricMSE(
        pred=pred_scalars_key,
        target=target_scalars_key,
        batch_pre_collect_process_func=process_func,
    )

    metrics[f"{name}_rmse"] = MetricRMSE(
        pred=pred_scalars_key,
        target=target_scalars_key,
        batch_pre_collect_process_func=process_func,
    )

    metrics[f"{name}_r2"] = MetricR2(
        pred=pred_scalars_key,
        target=target_scalars_key,
        batch_pre_collect_process_func=process_func,
    )

    return metrics

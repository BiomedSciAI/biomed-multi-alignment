"""
mammal_utils.py — Shared utilities for MAMMAL inference notebooks.

Provides:
  - InferenceDataset   : PyTorch Dataset for SMILES inference from a DataFrame
  - build_dataloader   : Build a DataLoader from a DataFrame
  - load_tokenizer     : Load the MAMMAL ModularTokenizerOp
  - load_model         : Load a finetuned MAMMAL Lightning checkpoint
  - run_inference      : Run the full inference loop and return a DataFrame
  - calculate_enrichment_metrics : Precision@K and EF@K
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
import torch

# ── MAMMAL / FuseMedML imports ─────────────────────────────────────────────────
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from fuse.data.utils.collates import CollateDefault
from fuse.dl.lightning.pl_module import LightningModuleDefault
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

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
from mammal.model import Mammal

# ── Silence noisy C-level output ──────────────────────────────────────────────


@contextlib.contextmanager
def _redirect_fds(stdout: bool = True, stderr: bool = True):
    """Temporarily redirect file descriptors 1/2 to /dev/null."""
    to_restore, temps = [], []
    try:
        if stdout:
            saved = os.dup(1)
            to_restore.append((1, saved))
            f = tempfile.TemporaryFile(mode="w+b")
            temps.append(f)
            os.dup2(f.fileno(), 1)
        if stderr:
            saved = os.dup(2)
            to_restore.append((2, saved))
            f = tempfile.TemporaryFile(mode="w+b")
            temps.append(f)
            os.dup2(f.fileno(), 2)
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        for fd, saved in reversed(to_restore):
            os.dup2(saved, fd)
            os.close(saved)
        for f in temps:
            try:
                f.close()
            except Exception:
                pass


# ── Dataset ────────────────────────────────────────────────────────────────────


class InferenceDataset(Dataset):
    """
    PyTorch Dataset for MAMMAL inference from a pandas DataFrame.

    Tokenises each SMILES on-the-fly and returns a sample dict ready for
    CollateDefault / model.generate().

    Args:
        df: Input DataFrame.
        smiles_column: Column name containing SMILES strings.
        label_column: Column name containing binary labels, or ``None`` if the
            file has no labels (pure inference mode).
        tokenizer_op: Loaded ``ModularTokenizerOp`` instance.
        drug_max_seq_length: Maximum SMILES token length passed to the tokeniser.
        encoder_input_max_seq_len: Maximum total encoder input length.
        labels_max_seq_len: Maximum label sequence length.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        smiles_column: str,
        label_column: str | None,
        tokenizer_op: ModularTokenizerOp,
        drug_max_seq_length: int = 300,
        encoder_input_max_seq_len: int = 320,
        labels_max_seq_len: int = 10,
    ) -> None:
        self.smiles = df[smiles_column].tolist()
        self.has_labels = label_column is not None and label_column in df.columns
        self.labels = df[label_column].tolist() if self.has_labels else [None] * len(df)
        self.indices = df.index.tolist()
        self.tokenizer_op = tokenizer_op
        self.drug_max_seq_length = drug_max_seq_length
        self.encoder_input_max_seq_len = encoder_input_max_seq_len
        self.labels_max_seq_len = labels_max_seq_len
        print(
            f"InferenceDataset: {len(self.smiles)} samples, has_labels={self.has_labels}"
        )

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict:
        drug_sequence = self.smiles[idx]
        label = self.labels[idx]

        sample: dict = {
            "index": self.indices[idx],
            "data.smiles": drug_sequence,
            "data.label": label,
        }

        # ── Encoder input ──────────────────────────────────────────────────────
        sample[ENCODER_INPUTS_STR] = (
            f"<@TOKENIZER-TYPE=SMILES><SENTINEL_ID_0>"
            f"<MOLECULAR_ENTITY><MOLECULAR_ENTITY_SMALL_MOLECULE>"
            f"<@TOKENIZER-TYPE=SMILES@MAX-LEN={self.drug_max_seq_length}>"
            f"<SEQUENCE_NATURAL_START>{drug_sequence}<SEQUENCE_NATURAL_END><EOS>"
        )
        self.tokenizer_op(
            sample_dict=sample,
            key_in=ENCODER_INPUTS_STR,
            key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
            key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK,
            max_seq_len=self.encoder_input_max_seq_len,
        )
        sample[ENCODER_INPUTS_TOKENS] = torch.tensor(sample[ENCODER_INPUTS_TOKENS])
        sample[ENCODER_INPUTS_ATTENTION_MASK] = torch.tensor(
            sample[ENCODER_INPUTS_ATTENTION_MASK]
        )

        # ── Labels / decoder input (only when labels are available) ────────────
        if label is not None:
            pad_id = self.tokenizer_op.get_token_id("<PAD>")
            ignore_token_value = -100

            sample[LABELS_STR] = (
                f"<@TOKENIZER-TYPE=SMILES><SENTINEL_ID_0><{label}><EOS>"
            )
            self.tokenizer_op(
                sample_dict=sample,
                key_in=LABELS_STR,
                key_out_tokens_ids=LABELS_TOKENS,
                key_out_attention_mask=LABELS_ATTENTION_MASK,
                max_seq_len=self.labels_max_seq_len,
            )
            sample[LABELS_TOKENS] = torch.tensor(sample[LABELS_TOKENS])
            sample[LABELS_ATTENTION_MASK] = torch.tensor(sample[LABELS_ATTENTION_MASK])
            # Replace PAD tokens with ignore value
            sample[LABELS_TOKENS][
                (sample[LABELS_TOKENS][..., None] == torch.tensor(pad_id))
                .any(-1)
                .nonzero()
            ] = ignore_token_value

            sample[DECODER_INPUTS_STR] = (
                f"<@TOKENIZER-TYPE=SMILES><DECODER_START><SENTINEL_ID_0><{label}><EOS>"
            )
            self.tokenizer_op(
                sample_dict=sample,
                key_in=DECODER_INPUTS_STR,
                key_out_tokens_ids=DECODER_INPUTS_TOKENS,
                key_out_attention_mask=DECODER_INPUTS_ATTENTION_MASK,
                max_seq_len=self.labels_max_seq_len,
            )
            sample[DECODER_INPUTS_TOKENS] = torch.tensor(sample[DECODER_INPUTS_TOKENS])
            sample[DECODER_INPUTS_ATTENTION_MASK] = torch.tensor(
                sample[DECODER_INPUTS_ATTENTION_MASK]
            )

        return sample


# ── DataLoader factory ─────────────────────────────────────────────────────────


def build_dataloader(
    df: pd.DataFrame,
    tokenizer_op: ModularTokenizerOp,
    smiles_column: str = "smiles",
    label_column: str | None = "label",
    batch_size: int = 128,
    drug_max_seq_length: int = 300,
    encoder_input_max_seq_len: int = 320,
    labels_max_seq_len: int = 10,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader for MAMMAL inference.

    Args:
        df: Input DataFrame with SMILES (and optionally labels).
        tokenizer_op: Loaded ``ModularTokenizerOp``.
        smiles_column: Column name for SMILES.
        label_column: Column name for labels, or ``None`` for label-free inference.
        batch_size: Batch size.
        drug_max_seq_length: Maximum SMILES token length.
        encoder_input_max_seq_len: Maximum encoder input length.
        labels_max_seq_len: Maximum label sequence length.
        num_workers: DataLoader worker count.

    Returns:
        Configured ``DataLoader``.
    """
    dataset = InferenceDataset(
        df=df,
        smiles_column=smiles_column,
        label_column=label_column,
        tokenizer_op=tokenizer_op,
        drug_max_seq_length=drug_max_seq_length,
        encoder_input_max_seq_len=encoder_input_max_seq_len,
        labels_max_seq_len=labels_max_seq_len,
    )

    pad_token_id = tokenizer_op.get_token_id("<PAD>")
    special_handlers = {
        ENCODER_INPUTS_TOKENS: partial(
            CollateDefault.crop_padding, pad_token_id=pad_token_id
        ),
        ENCODER_INPUTS_ATTENTION_MASK: partial(
            CollateDefault.crop_padding, pad_token_id=False
        ),
    }

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=CollateDefault(special_handlers_keys=special_handlers),
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"DataLoader ready: {len(dataset)} samples, {len(loader)} batches")
    return loader


# ── Model loading ──────────────────────────────────────────────────────────────


def load_tokenizer(
    base_model_path: str = "ibm/biomed.omics.bl.sm.ma-ted-458m",
) -> ModularTokenizerOp:
    """
    Load the MAMMAL ``ModularTokenizerOp``, suppressing noisy C-level output.

    Args:
        base_model_path: HuggingFace hub ID or local path to the base MAMMAL model.

    Returns:
        Loaded ``ModularTokenizerOp``.
    """
    with _redirect_fds(stdout=True, stderr=True):
        tokenizer_op = ModularTokenizerOp.from_pretrained(base_model_path)
    return tokenizer_op


def load_model(
    model_path: str,
    base_model_path: str = "ibm/biomed.omics.bl.sm.ma-ted-458m",
    device: str = "cpu",
) -> Mammal:
    """
    Load a finetuned MAMMAL model from a Lightning checkpoint.

    Args:
        model_path: Path to the ``.ckpt`` Lightning checkpoint.
        base_model_path: HuggingFace hub ID or local path to the base MAMMAL model.
        device: Device to move the model to after loading (``"cpu"`` or ``"cuda"``).

    Returns:
        The inner ``Mammal`` model in eval mode on the requested device.
    """
    base_model = Mammal.from_pretrained(base_model_path)
    pl_module = LightningModuleDefault.load_from_checkpoint(
        checkpoint_path=model_path,
        model_dir=None,
        model=base_model,
        map_location="cpu",
    )
    model = pl_module._model  # extract the inner Mammal
    model.eval().to(device)
    return model


# ── Inference loop ─────────────────────────────────────────────────────────────


def run_inference(
    model: Mammal,
    dataloader: DataLoader,
    tokenizer_op: ModularTokenizerOp,
    device: str = "cpu",
    classification_position: int = 1,
) -> pd.DataFrame:
    """
    Run the MAMMAL inference loop over a DataLoader.

    For each sample the function extracts:
    - ``predicted_label``  : 0 or 1 (argmax of <0>/<1> logits)
    - ``prediction_score`` : softmax probability for the positive class
    - ``raw_score_negative`` / ``raw_score_positive`` : raw logits

    Args:
        model: Loaded ``Mammal`` model (eval mode).
        dataloader: DataLoader produced by :func:`build_dataloader`.
        tokenizer_op: Loaded ``ModularTokenizerOp``.
        device: Device to run inference on.
        classification_position: Position in the decoder output that holds the
            class token (default ``1``).

    Returns:
        DataFrame with columns:
        ``sample_id``, ``smiles``, ``true_label``, ``predicted_label``,
        ``prediction_score``, ``raw_score_negative``, ``raw_score_positive``.
    """
    negative_token_id = tokenizer_op.get_token_id("<0>")
    positive_token_id = tokenizer_op.get_token_id("<1>")

    results: dict[str, list] = {
        "sample_id": [],
        "smiles": [],
        "true_label": [],
        "predicted_label": [],
        "prediction_score": [],
        "raw_score_negative": [],
        "raw_score_positive": [],
    }

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            batch_size = batch[ENCODER_INPUTS_TOKENS].shape[0]

            # Build per-sample dicts for model.generate()
            sample_dicts = []
            for i in range(batch_size):
                sd: dict = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        sd[key] = value[i].to(device)
                    elif isinstance(value, list):
                        sd[key] = value[i]
                    else:
                        sd[key] = value
                sample_dicts.append(sd)

            batch_out = model.generate(
                sample_dicts,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=5,
            )

            decoder_output = batch_out.get(CLS_PRED, None)  # (B, seq_len)
            decoder_scores = batch_out.get(SCORES, None)  # (B, seq_len, vocab)

            for i in range(batch_size):
                sample_id = batch.get("index", [None] * batch_size)[i]
                smiles = batch.get("data.smiles", [None] * batch_size)[i]
                true_label = batch.get("data.label", [None] * batch_size)[i]

                if isinstance(sample_id, torch.Tensor):
                    sample_id = sample_id.item()
                if isinstance(true_label, torch.Tensor):
                    true_label = true_label.item()

                # Initialize variables with Optional types
                predicted_label: int | None = None
                prediction_score: float | None = None
                raw_score_negative: float | None = None
                raw_score_positive: float | None = None

                if decoder_output is not None and decoder_scores is not None:
                    out_tokens = decoder_output[i].cpu().numpy()
                    out_scores = decoder_scores[i].cpu().numpy()  # (seq_len, vocab)

                    predicted_token = int(out_tokens[classification_position])
                    pos_score = float(
                        out_scores[classification_position, positive_token_id]
                    )
                    neg_score = float(
                        out_scores[classification_position, negative_token_id]
                    )

                    label_map = {negative_token_id: 0, positive_token_id: 1}
                    if predicted_token in label_map:
                        predicted_label = label_map[predicted_token]
                    else:
                        # Fallback: use scores
                        predicted_label = 1 if pos_score > neg_score else 0

                    prediction_score = pos_score
                    raw_score_negative = neg_score
                    raw_score_positive = pos_score

                results["sample_id"].append(sample_id)
                results["smiles"].append(smiles)
                results["true_label"].append(true_label)
                results["predicted_label"].append(predicted_label)
                results["prediction_score"].append(prediction_score)
                results["raw_score_negative"].append(raw_score_negative)
                results["raw_score_positive"].append(raw_score_positive)

    return pd.DataFrame(results)


# ── Enrichment metrics ─────────────────────────────────────────────────────────


def calculate_enrichment_metrics(
    y_true,
    y_scores,
    top_k_values: list[int] | None = None,
) -> dict:
    """
    Calculate Precision@K and Enrichment Factor (EF@K).

    ``EF@K = Precision@K / hit_prevalence``
    where ``hit_prevalence = fraction of positives in the full dataset``.

    Args:
        y_true: Array-like of true binary labels.
        y_scores: Array-like of predicted scores / probabilities.
        top_k_values: List of K values to evaluate.  Defaults to
            ``[10, 50, 100, 500, 1000, 2000]``.

    Returns:
        Dict with ``Precision@K`` and ``EF@K`` for each K.
    """
    if top_k_values is None:
        top_k_values = [10, 50, 100, 500, 1000, 2000]

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_labels = y_true[sorted_indices]

    hit_prevalence = np.sum(y_true) / len(y_true)

    enrichment: dict = {}
    for k in top_k_values:
        if k <= len(sorted_labels):
            top_k_labels = sorted_labels[:k]
            precision_at_k = np.sum(top_k_labels) / k
            enrichment[f"Precision@{k}"] = precision_at_k
            enrichment[f"EF@{k}"] = (
                precision_at_k / hit_prevalence if hit_prevalence > 0 else np.nan
            )
        else:
            enrichment[f"Precision@{k}"] = np.nan
            enrichment[f"EF@{k}"] = np.nan

    return enrichment


# Made with Bob

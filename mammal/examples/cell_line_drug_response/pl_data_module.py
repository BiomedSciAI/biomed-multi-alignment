"""
PyTorch Lightning DataModule for cell line drug response fine-tuning.

Handles data loading, preprocessing, and tokenization for MAMMAL model.
"""

from collections.abc import Callable

import pytorch_lightning as pl
from fuse.data import OpBase
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from fuse.data.utils.collates import CollateDefault
from fuse.utils import NDict
from torch.utils.data import DataLoader

from mammal.examples.cell_line_drug_response.dataset import GDSCDataset


class OpReadGDSCDataset(OpBase):

    def __init__(self, dataset: GDSCDataset):
        super().__init__()
        self.dataset = dataset

    def __call__(self, sample_dict: NDict, **kwargs) -> NDict:
        idx = sample_dict["data.sample_id"]
        sample = self.dataset[idx]
        sample_dict.update(sample)
        return sample_dict


class OpPreprocessSample(OpBase):

    def __init__(
        self,
        data_preprocessing: Callable,
        tokenizer_op: ModularTokenizerOp,
        encoder_input_max_seq_len: int,
    ):
        super().__init__()
        self.data_preprocessing = data_preprocessing
        self.tokenizer_op = tokenizer_op
        self.encoder_input_max_seq_len = encoder_input_max_seq_len

    def __call__(self, sample_dict: NDict, **kwargs) -> NDict:
        if not isinstance(sample_dict, NDict):
            sample_dict = NDict(sample_dict)

        result = self.data_preprocessing(
            sample_dict,
            genes_key="genes",
            expressions_key="expressions",
            drug_smiles_key="smiles",
            ground_truth_key="Y",
            tokenizer_op=self.tokenizer_op,
            encoder_input_max_seq_len=self.encoder_input_max_seq_len,
        )

        sample_dict.update(result)
        return sample_dict


class CellLineDrugResponseDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for cell line drug response fine-tuning."""

    def __init__(
        self,
        *,
        batch_size: int,
        tokenizer_op: ModularTokenizerOp,
        train_dl_kwargs: dict,
        valid_dl_kwargs: dict,
        seed: int,
        data_preprocessing: Callable,
        encoder_input_max_seq_len: int,
        dataset_name: str = "GDSC2",
        dataset_path: str | None = None,
        limit_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer_op = tokenizer_op
        self.data_preprocessing = data_preprocessing
        self.encoder_input_max_seq_len = encoder_input_max_seq_len
        self.batch_size = batch_size
        self.train_dl_kwargs = train_dl_kwargs
        self.valid_dl_kwargs = valid_dl_kwargs
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.limit_samples = limit_samples

    def setup(self, stage: str) -> None:
        train_ds = GDSCDataset(
            name=self.dataset_name,
            path=self.dataset_path,
            split="train",
            limit_samples=self.limit_samples,
        )
        valid_ds = GDSCDataset(
            name=self.dataset_name,
            path=self.dataset_path,
            split="valid",
            limit_samples=self.limit_samples,
        )
        test_ds = GDSCDataset(
            name=self.dataset_name,
            path=self.dataset_path,
            split="test",
            limit_samples=self.limit_samples,
        )

        self.ds_dict = {}
        for name, ds in [("train", train_ds), ("valid", valid_ds), ("test", test_ds)]:
            dynamic_pipeline = PipelineDefault(
                "cell_line_drug_response",
                [
                    # Step 1: Load data
                    (OpReadGDSCDataset(ds), dict()),
                    # Step 2: Preprocess and tokenize
                    (
                        OpPreprocessSample(
                            data_preprocessing=self.data_preprocessing,
                            tokenizer_op=self.tokenizer_op,
                            encoder_input_max_seq_len=self.encoder_input_max_seq_len,
                        ),
                        dict(),
                    ),
                ],
            )

            wrapped_ds = DatasetDefault(
                sample_ids=list(range(len(ds))),
                dynamic_pipeline=dynamic_pipeline,
            )
            wrapped_ds.create()
            self.ds_dict[name] = wrapped_ds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.ds_dict["train"],
            batch_size=self.batch_size,
            collate_fn=CollateDefault(add_to_batch_dict={"forward_mode": "encoder"}),
            shuffle=True,
            **self.train_dl_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_dict["valid"],
            batch_size=self.batch_size,
            collate_fn=CollateDefault(add_to_batch_dict={"forward_mode": "encoder"}),
            **self.valid_dl_kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_dict["test"],
            batch_size=self.batch_size,
            collate_fn=CollateDefault(add_to_batch_dict={"forward_mode": "encoder"}),
            **self.valid_dl_kwargs,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

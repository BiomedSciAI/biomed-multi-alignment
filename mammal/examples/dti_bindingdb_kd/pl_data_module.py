from collections.abc import Callable

import pytorch_lightning as pl
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from fuse.data.tokenizers.modular_tokenizer.special_tokens import special_wrap_input
from fuse.data.utils.collates import CollateDefault
from tdc.multi_pred.dti import DTI
from torch.utils.data.dataloader import DataLoader

from mammal.keys import *  # noqa


class DtiBindingdbKdDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        batch_size: int,
        tokenizer_op: ModularTokenizerOp,
        train_dl_kwargs: dict,
        valid_dl_kwargs: dict,
        seed: int,
        data_preprocessing: Callable,
        target_max_seq_length: int,
        drug_max_seq_length: int,
        encoder_input_max_seq_len: int,
        load_datasets_kwargs: dict,
    ) -> None:
        super().__init__()
        self.tokenizer_op = tokenizer_op
        self.target_max_seq_length = target_max_seq_length
        self.drug_max_seq_length = drug_max_seq_length
        self.encoder_input_max_seq_len = encoder_input_max_seq_len
        self.batch_size = batch_size
        self.train_dl_kwargs = train_dl_kwargs
        self.valid_dl_kwargs = valid_dl_kwargs
        self.seed = seed
        self.data_preprocessing = data_preprocessing
        self.load_datasets_kwargs = load_datasets_kwargs

        self.pad_token_id = self.tokenizer_op.get_token_id(special_wrap_input("PAD"))

    def setup(self, stage: str) -> None:
        self.ds_dict = load_datasets(**self.load_datasets_kwargs)

        task_pipeline = [
            (
                # Prepare the input string(s) in modular tokenizer input format
                self.data_preprocessing,
                dict(
                    target_sequence_key="Target",
                    drug_sequence_key="Drug",
                    ground_truth_key="Y",
                    tokenizer_op=self.tokenizer_op,
                    target_max_seq_length=self.target_max_seq_length,
                    drug_max_seq_length=self.drug_max_seq_length,
                    encoder_input_max_seq_len=self.encoder_input_max_seq_len,
                ),
            ),
        ]

        for ds in self.ds_dict.values():
            ds.dynamic_pipeline.extend(task_pipeline)

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            dataset=self.ds_dict["train"],
            batch_size=self.batch_size,
            collate_fn=CollateDefault(add_to_batch_dict={"forward_mode": "encoder"}),
            shuffle=True,
            **self.train_dl_kwargs,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            self.ds_dict["valid"],
            batch_size=self.batch_size,
            collate_fn=CollateDefault(add_to_batch_dict={"forward_mode": "encoder"}),
            **self.valid_dl_kwargs,
        )

        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(
            self.ds_dict["test"],
            batch_size=self.batch_size,
            collate_fn=CollateDefault(add_to_batch_dict={"forward_mode": "encoder"}),
            **self.valid_dl_kwargs,
        )

        return test_loader

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


def load_datasets(
    split_type: str = "cold_split", split_column: list[str] | str = ["Drug", "Target"]
) -> dict[str, DatasetDefault]:
    """
    Automatically downloads (using tdc) the data and create dataset iterator for "train", "val" and "test".
    :return: dictionary that maps fold name "train", "val" and "test" to a dataset iterator
    """

    data = DTI(name="BindingDB_Kd")
    data.harmonize_affinities(mode="max_affinity")
    data.convert_to_log(form="binding")
    split = data.get_split(method=split_type, column_name=split_column)
    ds_dict = {}
    for set_name in ["train", "valid", "test"]:
        set_df = split[set_name]
        print(f"{set_name} set size is {len(set_df)}")
        print(f"{set_name=} {set_df.Y.mean()=} {set_df.Y.std()=}")
        dynamic_pipeline = PipelineDefault(
            "dti",
            [
                (
                    OpReadDataframe(
                        set_df,
                        key_column=None,
                        columns_to_extract=["Target", "Drug", "Y"],
                    ),
                    dict(),
                ),
            ],
        )

        ds = DatasetDefault(sample_ids=len(set_df), dynamic_pipeline=dynamic_pipeline)
        ds.create()
        ds_dict[set_name] = ds

    return ds_dict


if __name__ == "__main__":
    load_datasets()

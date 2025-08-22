from collections.abc import Callable

import pytorch_lightning as pl
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from fuse.data.utils.collates import CollateDefault
from tdc.single_pred.tox import Tox
from torch.utils.data.dataloader import DataLoader

from mammal.keys import *  # noqa


class CarcinogenicityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        batch_size: int,
        tokenizer_op: ModularTokenizerOp,
        drug_max_seq_length: int,
        encoder_input_max_seq_len: int,
        data_preprocessing: Callable,
        labels_max_seq_len: int,
    ) -> None:
        super().__init__()
        self.tokenizer_op = tokenizer_op
        self.drug_max_seq_length = drug_max_seq_length
        self.encoder_input_max_seq_len = encoder_input_max_seq_len
        self.labels_max_seq_len = labels_max_seq_len
        self.batch_size = batch_size
        self.data_preprocessing = data_preprocessing
        self.pad_token_id = self.tokenizer_op.get_token_id("<PAD>")

    def setup(self, stage: str) -> None:
        self.ds_dict = load_datasets()

        task_pipeline = [
            (
                self.data_preprocessing,
                dict(
                    sequence_key="data.drug",
                    label_key="data.label",
                    tokenizer_op=self.tokenizer_op,
                    encoder_input_max_seq_len=self.encoder_input_max_seq_len,
                    labels_max_seq_len=self.labels_max_seq_len,
                ),
            ),
        ]

        for ds in self.ds_dict.values():
            ds.dynamic_pipeline.extend(task_pipeline)

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            dataset=self.ds_dict["train"],
            batch_size=self.batch_size,
            collate_fn=CollateDefault(),
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            self.ds_dict["valid"],
            batch_size=self.batch_size,
            collate_fn=CollateDefault(),
        )

        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(
            self.ds_dict["test"],
            batch_size=self.batch_size,
            collate_fn=CollateDefault(),
        )

        return test_loader

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


def load_datasets(split_method: str = "random") -> dict[str, DatasetDefault]:
    data = Tox(name="Carcinogens_Lagunin")
    split = data.get_split(method=split_method)

    ds_dict = {}
    for set_name in ["train", "valid", "test"]:
        data_df = split[set_name]
        print(f"{set_name} set size is {len(data_df)}")
        size = len(data_df)

        dynamic_pipeline = PipelineDefault(
            "carcinogenicity",
            [
                (
                    OpReadDataframe(
                        data_df,
                        key_column=None,
                        rename_columns={"Drug": "data.drug", "Y": "data.label"},
                    ),
                    dict(),
                ),
            ],
        )

        ds = DatasetDefault(sample_ids=size, dynamic_pipeline=dynamic_pipeline)
        ds.create()
        ds_dict[set_name] = ds

    return ds_dict


if __name__ == "__main__":
    ds = load_datasets()
    print(ds["train"][0])
    print(ds["test"][0])

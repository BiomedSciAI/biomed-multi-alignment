import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
from anndata_op import OpReadAnnData
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from fuse.data.utils.collates import CollateDefault
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader

from mammal.keys import *  # noqa


class CellTypeDataModule(pl.LightningDataModule):
    skip_keys = [
        "scrna.gene_names",
        "scrna.scrna",
    ]

    def __init__(
        self,
        *,
        data_path: str,
        batch_size: int,
        tokenizer_op: ModularTokenizerOp,
        data_preprocessing: Callable,
        train_dl_kwargs: dict,
        valid_dl_kwargs: dict,
        label_name="celltype",
        input_max_seq_length: int = 500,
        encoder_input_max_seq_len: int = 512,
        labels_max_seq_len: int = 20,
        seed: int = 42,
        stratify_by_label=False,
    ) -> None:
        """_summary_
        Args:
            data_path (str): path to the raw data, if not exist, will download the data to the given path.
            batch_size (int): batch size
            tokenizer_op (ModularTokenizerOp): tokenizer op
            encoder_inputs_max_seq_len: max tokenizer sequence length for the encoder inputs,
            labels_max_seq_len: max tokenizer sequence length for the labels,
            train_dl_kwargs (dict): train dataloader constructor parameters
            valid_dl_kwargs (dict): validation dataloader constructor parameters
            seed (int): random seed
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer_op = tokenizer_op
        self.input_max_seq_length = input_max_seq_length
        self.encoder_input_max_seq_len = encoder_input_max_seq_len
        self.labels_max_seq_len = labels_max_seq_len
        self.batch_size = batch_size
        self.train_dl_kwargs = train_dl_kwargs
        self.valid_dl_kwargs = valid_dl_kwargs
        self.label_name = label_name
        self.seed = seed
        self.data_preprocessing = data_preprocessing
        self.pad_token_id = self.tokenizer_op.get_token_id("<PAD>")
        self.ds_dict: dict[str, Any] = {}
        if stratify_by_label:
            self.stratify_by = self.label_name
        else:
            self.stratify_by = None

    def setup(self, stage: str) -> None:
        self.ds_dict = load_datasets(
            data_path=self.data_path, stratify_by=self.stratify_by
        )

        task_pipeline = [
            (
                # Prepare the input string(s) in modular tokenizer input format
                self.data_preprocessing,
                dict(
                    sequence_key="scrna.scrna",
                    label_key="data.label",
                    tokenizer_op=self.tokenizer_op,
                    input_max_seq_length=self.input_max_seq_length,
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
            collate_fn=self.collate_fn(),
            shuffle=True,
            **self.train_dl_kwargs,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            self.ds_dict["valid"],
            batch_size=self.batch_size,
            collate_fn=self.collate_fn(),
            **self.valid_dl_kwargs,
        )

        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(
            self.ds_dict["test"],
            batch_size=self.batch_size,
            collate_fn=self.collate_fn(),
            **self.valid_dl_kwargs,
        )

        return test_loader

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def collate_fn(self):
        return CollateDefault(skip_keys=self.skip_keys)


def anndata_train_test_split(
    h5ad_file, test_size=0.1, random_state=42, stratify_by=None
):
    # h5ad_file.obs = h5ad_file.obs.rename(
    # columns={"celltype": "label", "CellType": "label"}
    # )

    if stratify_by is not None:
        stratify = h5ad_file.obs[stratify_by]
    else:
        stratify = None

    train_ids, valid_ids = train_test_split(
        h5ad_file.obs,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_adata = h5ad_file[train_ids.index]
    validata_adata = h5ad_file[valid_ids.index]
    return train_adata, validata_adata


def load_datasets(
    data_path: str | Path = "data", stratify_by=None
) -> dict[str, DatasetDefault]:

    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent / data_path
    # read files
    anndata_object = anndata.read_h5ad(data_path)
    # Without this the anndata on file is assumed to be post preprocessing.
    # preprocess_ann_data(anndata_object)
    anndata_dict = {}
    anndata_dict["all_data"] = anndata_object
    anndata_dict["all_train"], anndata_dict["test"] = anndata_train_test_split(
        anndata_dict["all_data"],
        test_size=0.1,
        stratify_by=stratify_by,
    )
    anndata_dict["train"], anndata_dict["valid"] = anndata_train_test_split(
        anndata_dict["all_train"],
        test_size=0.1 / (1.0 - 0.1),
        random_state=2024,
        stratify_by=stratify_by,
    )

    ds_dict = {}
    for set_name in ["train", "valid", "test"]:
        input_anndata = anndata_dict[set_name]
        size = input_anndata.shape[0]
        print(f"{set_name} set size is {size}")

        dynamic_pipeline = PipelineDefault(
            "cell_type",
            [
                (OpReadAnnData(input_anndata), {"prefix": "scrna"}),
            ],
        )

        ds = DatasetDefault(sample_ids=size, dynamic_pipeline=dynamic_pipeline)
        ds.create()
        ds_dict[set_name] = ds

    return ds_dict


def load_cell_type_mapping(
    mapping_key="celltype", mapping_value="cell_type_ontology_term_id"
):
    """
    Load metadata_extra_mapping.csv from the given dataset metadata folder,
    and return the values of a requested key and value columns as a dictionary.

    This is used to convert the names from the ones in the input anndata to the
    ones that are known to the tokenizer.
    """
    cell_type_mapping_file_path = Path(__file__).parent / "cell_type_mapping.csv"

    #    this is new
    print(cell_type_mapping_file_path)
    if not os.path.exists(cell_type_mapping_file_path):
        raise FileNotFoundError(str(cell_type_mapping_file_path) + "is not found")
    else:
        mapping_df = pd.read_csv(cell_type_mapping_file_path, index_col=False)
        cell_type_mapping = dict(
            zip(
                mapping_df[mapping_key],
                mapping_df[mapping_value],
            )
        )
        return cell_type_mapping


def preprocess_ann_data(
    anndata_object: anndata.AnnData,
    min_genes: int = 200,
    normalize_total: float = 1000,
    num_bins: int = 10,
):
    """run preprocessing steps on anndata object
    assumes that the anndata object has a standard structure with counts per cell X gene, and cell type annotations in obs["celltype"].

    steps include:
        - translate cell types to ontology term ids
        - filter out cells with less than 200 genes expressed
        - normalize expression data sum to 1000
        - transform counts via log1p in base 2
        - digitize expression data into bins

    Args:
        ann_data_object (anndata.AnnData): input object.  will be overwritten
    """
    cell_type_mapper = load_cell_type_mapping()

    # place labels (cell types) in the "label" observations
    anndata_object.obs["label"] = [
        cell_type_mapper[cell] for cell in anndata_object.obs["celltype"]
    ]

    # process the data:

    # filter out cells with shallow reads
    sc.pp.filter_cells(anndata_object, min_genes=min_genes)
    # normalize depth and
    sc.pp.normalize_total(anndata_object, target_sum=normalize_total)
    # change to log1p space, which is approximately in the scale of 0 to 10 (log_2(1001)~=10)
    sc.pp.log1p(anndata_object, base=2)

    # split range to bins - more or less 0,2,3,..10 if using about  10 bins
    # the +1 is intended to create num_bin bins, as `linespace` creates the bin ends starting at the lowest and ending at the highest values.

    # find bin ends
    bins = np.linspace(
        anndata_object.X.data.min(), anndata_object.X.max(), num=num_bins + 1
    )

    # convert the counts to bins
    anndata_object.X.data = np.digitize(anndata_object.X.data, bins)

    return anndata_object


if __name__ == "__main__":
    _ds_dict = load_datasets("data/Zheng68k_filtered.h5ad", stratify_by=["label"])
    t0 = _ds_dict["train"][0]
    print(t0)
    print(_ds_dict["test"][0])

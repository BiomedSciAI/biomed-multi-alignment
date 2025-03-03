import os

import anndata
import click
import torch
from anndata_op import OpReadAnnData
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from fuse.data.utils.collates import CollateDefault
from pl_data_module import CellTypeDataModule
from task import CellTypeTask
from torch.utils.data.dataset import Dataset

from mammal.keys import (
    CLS_PRED,
    SCORES,
)
from mammal.model import Mammal


@click.command()
@click.argument("task_name", default="cell_type")
@click.option(
    "--model-path",
    default="/dccstor/mm_hcls/usr/matanin/mammal_extention/scrna_cell_type/mammalian_scRNA_cell_type_zeng68_24h.ckpt",
    help="Specify the model dir.",
)
@click.option(
    "--tokenizer_path",
    default="/dccstor/mm_hcls/usr/matanin/mammal_extention/scrna_cell_type/",
    help="Specify the tokenizer path.",
)
@click.option(
    "--h5ad-file-path",
    "-i",
    type=str,
    help="Specify the A5HD (AnnData) input file.",
    default="/u/matanin/git/biomed-multi-alignment/mammal/examples/scrna_cell_type/data/Zheng_68k_processed.h5ad",
)
@click.option("--sample_id", "-s", type=int, default=0)
@click.option(
    "--device", default="cpu", help="Specify the device to use (default: 'cpu')."
)
def main(
    task_name: str,
    h5ad_file_path: str,
    sample_id: int,
    model_path: str,
    tokenizer_path: str,
    device: str,
):

    tokenizer_op, nn_model = get_tokenizer_and_model(tokenizer_path, model_path, device)
    # task_dict = load_model(task_name=task_name, device=device)
    # convert to MAMMAL style

    anndata_object = anndata.read_h5ad(h5ad_file_path)

    dynamic_pipeline = PipelineDefault(
        "cell_type",
        [
            (OpReadAnnData(data=anndata_object), {"prefix": "scrna"}),
        ],
    )

    data_source = DatasetDefault(
        sample_ids=anndata_object.shape[0], dynamic_pipeline=dynamic_pipeline
    )
    data_source.create()

    sample_dict = create_sample_dict(task_name, data_source, sample_id, tokenizer_op)

    batch_dict = get_predictions(nn_model, sample_dict)

    ans = process_model_output(tokenizer_op, batch_dict)
    ans = {
        k: v.detach().numpy() if isinstance(v, torch.Tensor) else v
        for k, v in ans.items()
    }
    print(ans)


def process_model_output(tokenizer_op, batch_dict):
    return CellTypeTask.process_model_output(
        tokenizer_op=tokenizer_op,
        decoder_output=batch_dict[CLS_PRED][0],
        decoder_output_scores=batch_dict[SCORES][0],
    )


def get_tokenizer_and_model(tokenizer_path, model_path, device):
    tokenizer_op = ModularTokenizerOp.from_pretrained(
        os.path.join(tokenizer_path, "tokenizer")
    )
    nn_model = Mammal.from_pretrained(
        pretrained_model_name_or_path=model_path,
    )
    nn_model.eval()
    nn_model.to(device=device)

    return tokenizer_op, nn_model


def create_sample_dict(
    task_name: str, data_source: Dataset, sample_id: int, tokenizer_op
):
    sequence_key = "scrna.scrna"
    # Create and load sample
    sample_dict = data_source[sample_id]

    sample_dict = CellTypeTask.data_preprocessing(
        sample_dict,
        sequence_key=sequence_key,
        input_max_seq_length=1260,
        encoder_input_max_seq_len=1260,
        labels_max_seq_len=4,
        tokenizer_op=tokenizer_op,
    )

    return sample_dict


def get_predictions(model, sample_dict):
    batch_dict = CollateDefault(skip_keys=CellTypeDataModule.skip_keys)([sample_dict])
    return model.generate(
        batch_dict,
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=5,
    )


def print_result(batch_dict, scalars_preds_processed_key):

    value = batch_dict[scalars_preds_processed_key]
    ans = {"scalar_result": value}

    # Print prediction

    batch_dict["scalar_result"] = value
    print(f"estimated value: {ans}")


def process_sample(tokenizer_op, nn_model, sample_dict):
    # running in generate mode
    batch_dict = nn_model.generate(
        [sample_dict],
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=5,
    )

    return batch_dict


if __name__ == "__main__":
    main()

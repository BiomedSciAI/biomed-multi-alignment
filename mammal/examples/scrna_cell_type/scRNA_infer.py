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

    # module(
    #     task=task,
    #     model=model,
    #     **OmegaConf.to_container(cfg.module, resolve=True),
    # )

    sample_dict = create_sample_dict(task_name, data_source, sample_id, tokenizer_op)

    # result = task_infer(task_dict=task_dict, smiles_seq=sample_id)
    batch_dict = get_predictions(nn_model, sample_dict)
    # running in generate mode

    # batch_dict = process_sample(
    #     tokenizer_op, nn_model, sample_dict
    # )

    ans = process_model_output(tokenizer_op, batch_dict)
    # print(f"The prediction for {sample_id=} is {result}")
    # print_result(batch_dict, scalars_preds_processed_key)
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


# def load_model(task_name: str, device: str) -> dict:

#     # path = "ibm/biomed.omics.bl.sm.ma-ted-458m"
#     path = "/dccstor/mm_hcls/usr/matanin/mammal_extention/scrna_cell_type/mammalian_scRNA_cell_type_zeng68_24h.ckpt"

#     # Load Model and set to evaluation mode
#     model = Mammal.from_pretrained(path)
#     model.eval()
#     model.to(device=device)

#     # Load Tokenizer
#     tokenizer_op = ModularTokenizerOp.from_pretrained(path)

#     task_dict = dict(
#         task_name=task_name,
#         model=model,
#         tokenizer_op=tokenizer_op,
#     )
#     return task_dict


# def process_model_output(
#     tokenizer_op: ModularTokenizerOp,
#     decoder_output: np.ndarray,
#     decoder_output_scores: np.ndarray,
# ) -> dict | None:
#     """
#     Extract predicted class and scores
#     """
#     return CellTypeTask.process_model_output(
#         tokenizer_op,
#         decoder_output=decoder_output,
#         decoder_output_scores=decoder_output_scores,
#     )
#     # negative_token_id = tokenizer_op.get_token_id("<0>")
#     # positive_token_id = tokenizer_op.get_token_id("<1>")
#     # label_id_to_int = {
#     #     negative_token_id: 0,
#     #     positive_token_id: 1,
#     # }
#     # classification_position = 1

#     # if decoder_output_scores is not None:
#     #     scores = decoder_output_scores[classification_position, positive_token_id]

#     # ans = dict(
#     #     pred=label_id_to_int.get(int(decoder_output[classification_position]), -1),
#     #     score=scores.item(),
#     # )
#     # return ans


# def task_infer(task_dict: dict, smiles_seq: str) -> dict | None:
#     task_name = task_dict["task_name"]
#     model = task_dict["model"]
#     tokenizer_op = task_dict["tokenizer_op"]

#     sample_dict = create_sample_dict(task_name, smiles_seq, tokenizer_op)
#     # Generate Prediction
#     batch_dict = get_predictions(model, sample_dict)

#     # Post-process the model's output
#     result = process_model_output(
#         tokenizer_op=tokenizer_op,
#         decoder_output=batch_dict[CLS_PRED][0],
#         decoder_output_scores=batch_dict[SCORES][0],
#     )
#     return result


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
    # {
    #     sequence_key: data_source[sample_id],
    #     SAMPLE_ID: sample_id,
    # }

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
    # for k, v in batch_dict.items():
    #     if "model" in k:
    #         print(f"{k}: {v}")


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

import click
import numpy as np
import torch
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

from mammal.keys import (
    CLS_PRED,
    ENCODER_INPUTS_ATTENTION_MASK,
    ENCODER_INPUTS_STR,
    ENCODER_INPUTS_TOKENS,
    SCORES,
)
from mammal.model import Mammal

TASK_NAMES = ["BBBP", "TOXICITY", "FDA_APPR"]


@click.command()
@click.argument("task_name", default="BBBP")
@click.argument(
    "smiles_seq",
    default="C(Cl)Cl",
)
@click.option(
    "--device", default="cpu", help="Specify the device to use (default: 'cpu')."
)
def main(task_name: str, smiles_seq: str, device: str):
    task_dict = load_model(task_name=task_name, device=device)
    result = task_infer(task_dict=task_dict, smiles_seq=smiles_seq)
    print(f"The prediction for {smiles_seq=} is {result}")


def load_model(task_name: str, device: str) -> dict:
    match task_name:
        case "BBBP":
            path = "ibm/biomed.omics.bl.sm.ma-ted-458m.moleculenet_bbbp"
        case "TOXICITY":
            path = "ibm/biomed.omics.bl.sm.ma-ted-458m.moleculenet_clintox_tox"
        case "FDA_APPR":
            path = "ibm/biomed.omics.bl.sm.ma-ted-458m.moleculenet_clintox_fda"
        case _:
            print(f"The {task_name=} is incorrect")

    # Load Model and set to evaluation mode
    model = Mammal.from_pretrained(path)
    model.eval()
    model.to(device=device)

    # Load Tokenizer
    tokenizer_op = ModularTokenizerOp.from_pretrained(path)

    task_dict = dict(
        task_name=task_name,
        model=model,
        tokenizer_op=tokenizer_op,
    )
    return task_dict


def process_model_output(
    tokenizer_op: ModularTokenizerOp,
    decoder_output: np.ndarray,
    decoder_output_scores: np.ndarray,
) -> dict:
    """
    Extract predicted class and scores
    """
    negative_token_id = tokenizer_op.get_token_id("<0>")
    positive_token_id = tokenizer_op.get_token_id("<1>")
    label_id_to_int = {
        negative_token_id: 0,
        positive_token_id: 1,
    }
    classification_position = 1

    if decoder_output_scores is not None:
        scores = decoder_output_scores[classification_position, positive_token_id]

    ans = dict(
        pred=label_id_to_int.get(int(decoder_output[classification_position]), -1),
        score=scores.item(),
    )
    return ans


def create_sample_dict(task_name, smiles_seq, tokenizer_op, model):

    # Create and load sample
    sample_dict = dict()
    # Formatting prompt to match pre-training syntax
    sample_dict[ENCODER_INPUTS_STR] = (
        f"<@TOKENIZER-TYPE=SMILES><MOLECULAR_ENTITY><MOLECULAR_ENTITY_SMALL_MOLECULE><{task_name}><SENTINEL_ID_0><@TOKENIZER-TYPE=SMILES@MAX-LEN=2100><SEQUENCE_NATURAL_START>{smiles_seq}<SEQUENCE_NATURAL_END><EOS>"
    )

    # Tokenize
    tokenizer_op(
        sample_dict=sample_dict,
        key_in=ENCODER_INPUTS_STR,
        key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
        key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK,
    )
    sample_dict[ENCODER_INPUTS_TOKENS] = torch.tensor(
        sample_dict[ENCODER_INPUTS_TOKENS], device=model.device
    )
    sample_dict[ENCODER_INPUTS_ATTENTION_MASK] = torch.tensor(
        sample_dict[ENCODER_INPUTS_ATTENTION_MASK], device=model.device
    )
    return sample_dict


def get_batch_dict(model, sample_dict):
    return model.generate(
        [sample_dict],
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=5,
    )


def task_infer(task_dict: dict, smiles_seq: str) -> dict:
    task_name = task_dict["task_name"]
    model = task_dict["model"]
    tokenizer_op = task_dict["tokenizer_op"]

    if task_name not in TASK_NAMES:
        print(f"The {task_name=} is incorrect. Valid names are {TASK_NAMES}")

    sample_dict = create_sample_dict(task_name, smiles_seq, tokenizer_op, model)
    # Generate Prediction
    batch_dict = get_batch_dict(model, sample_dict)

    # Post-process the model's output
    result = process_model_output(
        tokenizer_op=tokenizer_op,
        decoder_output=batch_dict[CLS_PRED][0],
        decoder_output_scores=batch_dict[SCORES][0],
    )
    return result


if __name__ == "__main__":
    main()

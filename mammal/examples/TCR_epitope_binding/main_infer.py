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
    SAMPLE_ID,
)
from mammal.model import Mammal

TASK_NAMES = ["TCR_epitope_bind"]


@click.command()
@click.argument("task_name", default="TCR_epitope_bind")
@click.argument(
    "TCR_beta_seq",
    default="GAVVSQHPSWVICKSGTSVKIECRSLDFQATTMFWYRQFPKQSLMLMATSNEGSKATYEQGVEKDKFLINHASLTLSTLTVTSAHPEDSSFYICSASEGTSSYEQYFGPGTRLTVT",  #NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSYSWDRVLEQYFGPGTRLTVT
)
@click.argument(
    "epitope_seq",
    default="FLKEKGGL", #LLQTGIHVRVSQPSL
)
@click.option(
    "--device", default="cpu", help="Specify the device to use (default: 'cpu')."
)
def main(TCR_beta_seq: str, epitope_seq: str, device: str):
    task_name = "TCR_epitope_bind"
    task_dict = load_model(task_name=task_name, device=device)
    result = task_infer(task_dict=task_dict, TCR_beta_seq=TCR_beta_seq, epitope_seq=epitope_seq)
    print(f"The prediction for {epitope_seq} and {TCR_beta_seq} is {result}")


def load_model(device: str, task_name: str = "TCR_epitope_bind") -> dict:
    # path = "ibm/biomed.omics.bl.sm.ma-ted-458m" #change to "ibm/biomed.omics.bl.sm.ma-ted-458m.tcr_epitope_bind"
    path = "ibm/biomed.omics.bl.sm.ma-ted-458m.tcr_epitope_bind"

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


def task_infer(task_dict: dict, TCR_beta_seq: str, epitope_seq: str) -> dict:
    task_name = task_dict["task_name"]
    model = task_dict["model"]
    tokenizer_op = task_dict["tokenizer_op"]
    treat_inputs_as_general_proteins = False

    if task_name not in TASK_NAMES:
        print(f"The {task_name=} is incorrect. Valid names are {TASK_NAMES}")

    # Create and load sample
    sample_dict = dict()
    # Formatting prompt to match pre-training syntax
    
    
    if treat_inputs_as_general_proteins:
        # Treat inputs as general proteins:
        sample_dict[ENCODER_INPUTS_STR] = (
            f"<@TOKENIZER-TYPE=AA><BINDING_AFFINITY_CLASS><SENTINEL_ID_0><@TOKENIZER-TYPE=AA><MOLECULAR_ENTITY><MOLECULAR_ENTITY_GENERAL_PROTEIN><SEQUENCE_NATURAL_START>{TCR_beta_seq}<SEQUENCE_NATURAL_END><@TOKENIZER-TYPE=AA><MOLECULAR_ENTITY><MOLECULAR_ENTITY_GENERAL_PROTEIN><SEQUENCE_NATURAL_START>{epitope_seq}<SEQUENCE_NATURAL_END><EOS>"
        )
    else:
        # Treat inputs as TCR beta chain and epitope
        sample_dict[ENCODER_INPUTS_STR] = (
            f"<@TOKENIZER-TYPE=AA><BINDING_AFFINITY_CLASS><SENTINEL_ID_0><@TOKENIZER-TYPE=AA><MOLECULAR_ENTITY><MOLECULAR_ENTITY_TCR_BETA_VDJ><SEQUENCE_NATURAL_START>{TCR_beta_seq}<SEQUENCE_NATURAL_END><@TOKENIZER-TYPE=AA><MOLECULAR_ENTITY><MOLECULAR_ENTITY_EPITOPE><SEQUENCE_NATURAL_START>{epitope_seq}<SEQUENCE_NATURAL_END><EOS>"
        )
    
    
    
    sample_dict[SAMPLE_ID] = '1'

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

    # Generate Prediction
    batch_dict = model.generate(
        [sample_dict],
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=5,
    )

    # Post-process the model's output
    result = process_model_output(
        tokenizer_op=tokenizer_op,
        decoder_output=batch_dict[CLS_PRED][0],
        decoder_output_scores=batch_dict[SCORES][0],
    )
    return result


if __name__ == "__main__":
    main()
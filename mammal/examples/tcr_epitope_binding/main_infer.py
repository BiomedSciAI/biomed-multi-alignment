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


@click.command()
@click.argument(
    "tcr_beta_seq",
    default="GAVVSQHPSWVICKSGTSVKIECRSLDFQATTMFWYRQFPKQSLMLMATSNEGSKATYEQGVEKDKFLINHASLTLSTLTVTSAHPEDSSFYICSASEGTSSYEQYFGPGTRLTVT",  # alternative binder 1
    # Alternative binder 2: NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSYSWDRVLEQYFGPGTRLTVT
)
@click.argument(
    "epitope_seq",
    default="FLKEKGGL",  # alternative binder 1
    # Alternative binder 2: LLQTGIHVRVSQPSL
)
@click.option(
    "--device", default="cpu", help="Specify the device to use (default: 'cpu')."
)
def main(tcr_beta_seq: str, epitope_seq: str, device: str):
    model, tokenizer_op = load_model(device=device)
    result = task_infer(
        model=model,
        tokenizer_op=tokenizer_op,
        tcr_beta_seq=tcr_beta_seq,
        epitope_seq=epitope_seq,
    )
    print(f"The prediction for {epitope_seq} and {tcr_beta_seq} is {result}")


def load_model(
    device: str, path: str | None = None
) -> tuple["Mammal", "ModularTokenizerOp"]:
    if not path:
        path = "ibm/biomed.omics.bl.sm.ma-ted-458m.tcr_epitope_bind"  # change to "ibm/biomed.omics.bl.sm.ma-ted-458m" to try on the base model

    # Load Model and set to evaluation mode
    model = Mammal.from_pretrained(pretrained_model_name_or_path=path)
    model.eval()
    model.to(device=device)

    # Load Tokenizer
    tokenizer_op = ModularTokenizerOp.from_pretrained(path)

    return model, tokenizer_op


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


def task_infer(
    model: "Mammal",
    tokenizer_op: ModularTokenizerOp,
    tcr_beta_seq: str,
    epitope_seq: str,
) -> dict:
    treat_inputs_as_general_proteins = False

    # Create and load sample
    sample_dict = dict()
    # Formatting prompt to match pre-training syntax

    if treat_inputs_as_general_proteins:
        # Treat inputs as general proteins:
        sample_dict[ENCODER_INPUTS_STR] = (
            f"<@TOKENIZER-TYPE=AA><BINDING_AFFINITY_CLASS><SENTINEL_ID_0><@TOKENIZER-TYPE=AA><MOLECULAR_ENTITY><MOLECULAR_ENTITY_GENERAL_PROTEIN><SEQUENCE_NATURAL_START>{tcr_beta_seq}<SEQUENCE_NATURAL_END><@TOKENIZER-TYPE=AA><MOLECULAR_ENTITY><MOLECULAR_ENTITY_GENERAL_PROTEIN><SEQUENCE_NATURAL_START>{epitope_seq}<SEQUENCE_NATURAL_END><EOS>"
        )
    else:
        # Treat inputs as TCR beta chain and epitope
        sample_dict[ENCODER_INPUTS_STR] = (
            f"<@TOKENIZER-TYPE=AA><BINDING_AFFINITY_CLASS><SENTINEL_ID_0><@TOKENIZER-TYPE=AA><MOLECULAR_ENTITY><MOLECULAR_ENTITY_TCR_BETA_VDJ><SEQUENCE_NATURAL_START>{tcr_beta_seq}<SEQUENCE_NATURAL_END><@TOKENIZER-TYPE=AA><MOLECULAR_ENTITY><MOLECULAR_ENTITY_EPITOPE><SEQUENCE_NATURAL_START>{epitope_seq}<SEQUENCE_NATURAL_END><EOS>"
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

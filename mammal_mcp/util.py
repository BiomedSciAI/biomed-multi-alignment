import numpy as np
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp


def process_model_output(
    tokenizer_op: ModularTokenizerOp,
    decoder_output: np.ndarray,
    decoder_output_scores: np.ndarray,
) -> dict:
    """
    Extract predicted solubility class and scores
    expecting decoder output to be <SENTINEL_ID_0><0><EOS> or <SENTINEL_ID_0><1><EOS>
    note - the normalized version will calculate the positive ('<1>') score divided by the sum of the scores for both '<0>' and '<1>'
        BE CAREFUL as both negative and positive absolute scores can be drastically low, and normalized score could be very high.
    outputs a dictionary containing:
        dict(
            predicted_token_str = #... e.g. '<1>'
            not_normalized_score = #the score for the positive token... e.g.  0.01
            normalized_score = #... (positive_token_score) / (positive_token_score+negative_token_score)
        )
        if there is any error in parsing the model output, None is returned.
    """

    negative_token_id = tokenizer_op.get_token_id("<0>")
    positive_token_id = tokenizer_op.get_token_id("<1>")
    label_id_to_int = {
        negative_token_id: 0,
        positive_token_id: 1,
    }
    classification_position = 1

    if decoder_output_scores is not None:
        not_normalized_score = decoder_output_scores[
            classification_position, positive_token_id
        ]
        normalized_score = not_normalized_score / (
            not_normalized_score
            + decoder_output_scores[classification_position, negative_token_id]
            + 1e-10
        )
    ans = dict(
        pred=label_id_to_int.get(int(decoder_output[classification_position]), -1),
        not_normalized_scores=not_normalized_score,
        normalized_scores=normalized_score,
    )

    return ans

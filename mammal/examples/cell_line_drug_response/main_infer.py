import click
import numpy as np
import scanpy as sc
import torch
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from scipy.sparse import issparse

from mammal.keys import (
    ENCODER_INPUTS_ATTENTION_MASK,
    ENCODER_INPUTS_SCALARS,
    ENCODER_INPUTS_STR,
    ENCODER_INPUTS_TOKENS,
    SCALARS_PREDICTION_HEAD_LOGITS,
)
from mammal.model import Mammal


@click.command()
@click.option(
    "--model_path",
    required=True,
    help="Path to fine-tuned model directory",
)
@click.option(
    "--cell_line_h5ad_file",
    required=True,
    help="Path to h5ad file containing gene expression data",
)
@click.option(
    "--drug_smiles",
    required=True,
    help="SMILES representation of the drug",
)
@click.option(
    "--cell_line_name",
    default=None,
    help="Name of the cell line (optional, for display only)",
)
@click.option(
    "--drug_name",
    default=None,
    help="Name/ID of the drug (optional, for display only)",
)
@click.option(
    "--device",
    default="cpu",
    help="Device to use for inference (default: 'cpu')",
)
def main(
    model_path: str,
    cell_line_h5ad_file: str,
    drug_smiles: str,
    cell_line_name: str | None,
    drug_name: str | None,
    device: str,
):
    """
    Perform inference for a single cell line and drug combination.

    Example:
        python main_infer.py --model_path /path/to/model
        --cell_line_h5ad_file /path/to/A549.h5ad
        --drug_smiles "CC(=O)NCCC1=CNc2c1cc(OC)cc2"
        --cell_line_name "A549" --drug_name "Melatonin"
    """
    # Load Model
    model = Mammal.from_pretrained(model_path)
    model.eval()
    model.to(device=device)

    # Load Tokenizer
    tokenizer_op = ModularTokenizerOp.from_pretrained(
        "ibm/biomed.omics.bl.sm.ma-ted-458m"
    )

    print("=" * 80)
    print("Model and tokenizer loaded successfully.")
    print("=" * 80)

    # Run inference
    prediction = cell_line_drug_infer(
        model=model,
        tokenizer_op=tokenizer_op,
        h5ad_file=cell_line_h5ad_file,
        drug_smiles=drug_smiles,
        device=device,
    )

    # Print result
    print("\n" + "=" * 80)
    print("PREDICTION RESULT:")
    print("=" * 80)
    if cell_line_name:
        print(f"Cell Line: {cell_line_name}")
    if drug_name:
        print(f"Drug: {drug_name}")
    print(f"Predicted IC50: {prediction:.6f}")
    print("=" * 80)


def sort_genes_by_value_and_name(genes, expressions):
    """
    sort by both the expression value (as int in reverse order) and names in lexical order.
    """
    joined_vals = zip(expressions, genes)
    sorted_list = sorted(joined_vals, key=lambda x: (-int(x[0]), x[1]))
    return [gene for (_, gene) in sorted_list]


def cell_line_drug_infer(
    model,
    tokenizer_op,
    h5ad_file: str,
    drug_smiles: str,
    device: str = "cpu",
):
    """
    Predicts IC50 of a given drug over a given cell-line gene expression using MAMMAL a fine-tuned model checkpoint.

    :param model: Pre-loaded MAMMAL model
    :param tokenizer_op: Pre-loaded tokenizer
    :param h5ad_file: Path to h5ad file containing gene expression data
    :param drug_smiles: SMILES representation of the drug
    :param device: Device to use for inference
    :return: Prediction value
    """
    # Configuration parameters (matching training config)
    encoder_inputs_max_seq_len = 1500
    truncation_offset = 200
    format_length = 5
    max_genes = encoder_inputs_max_seq_len - truncation_offset - format_length

    # Read h5ad file
    adata = sc.read_h5ad(h5ad_file)

    # Extract gene names and expression values
    genes = adata.var_names.tolist()

    # Use the first cell's expression
    if adata.n_obs > 0:
        # Handle sparse matrices
        if issparse(adata.X):
            expressions = np.array(adata.X[0].todense()).flatten()
        else:
            expressions = np.array(adata.X[0]).flatten()
    else:
        raise ValueError(f"No cells found in {h5ad_file}")

    # Sort and truncate genes
    gene_seq = sort_genes_by_value_and_name(genes, expressions)[:max_genes]
    gene_seq_formatted = [f"[{gene}]" for gene in gene_seq]

    # Prepare Input Prompt
    sample_dict = dict()
    sample_dict[ENCODER_INPUTS_STR] = (
        "<@TOKENIZER-TYPE=SMILES><MASK>"
        + f"<@TOKENIZER-TYPE=SMILES><MOLECULAR_ENTITY><MOLECULAR_ENTITY_SMALL_MOLECULE><SMILES_SEQUENCE>{drug_smiles}"
        + "<@TOKENIZER-TYPE=GENE><MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>"
        + "".join(gene_seq_formatted)
        + "<EOS>"
    )

    # Tokenize
    tokenizer_op(
        sample_dict=sample_dict,
        key_in=ENCODER_INPUTS_STR,
        key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
        key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK,
        key_out_scalars=ENCODER_INPUTS_SCALARS,
        max_seq_len=encoder_inputs_max_seq_len,
        on_unknown="warn",
        verbose=0,
    )

    sample_dict[ENCODER_INPUTS_TOKENS] = torch.tensor(
        sample_dict[ENCODER_INPUTS_TOKENS]
    ).to(device)
    sample_dict[ENCODER_INPUTS_ATTENTION_MASK] = torch.tensor(
        sample_dict[ENCODER_INPUTS_ATTENTION_MASK]
    ).to(device)

    # Forward pass
    batch_dict = model.forward_encoder_only([sample_dict])

    # Extract prediction - get the first scalar prediction
    if SCALARS_PREDICTION_HEAD_LOGITS in batch_dict:
        predictions_full = batch_dict[SCALARS_PREDICTION_HEAD_LOGITS][0]
        prediction = predictions_full[0].squeeze().item()
        return prediction
    else:
        print("WARNING: No prediction found")
        return None


if __name__ == "__main__":
    main()

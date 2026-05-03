import anndata as ad
import click
import numpy as np
import scanpy as sc
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from scipy.sparse import issparse

from mammal.examples.cell_line_drug_response.task import CellLineDrugResponseTask
from mammal.keys import SCALARS_PREDICTION_HEAD_LOGITS
from mammal.model import Mammal


@click.command()
@click.option(
    "--model_path",
    required=True,
    help="Path to fine-tuned model directory",
)
@click.option(
    "--cell_line_h5ad_file",
    default=None,
    help="Path to h5ad file containing gene expression data",
)
@click.option(
    "--cell_line_name",
    default=None,
    help="Name of the cell line to load from tdc/GDSC2, Alternative to --cell_line_h5ad_file",
)
@click.option(
    "--drug_smiles",
    required=True,
    help="SMILES representation of the drug",
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
    cell_line_h5ad_file: str | None,
    cell_line_name: str | None,
    drug_smiles: str,
    drug_name: str | None,
    device: str,
):
    """
    Perform inference for a single cell line and drug combination.

    Examples:
        # Using h5ad file
        python main_infer.py --model_path /path/to/model \
            --cell_line_h5ad_file /path/to/A549.h5ad \
            --drug_smiles "CC(=O)NCCC1=CNc2c1cc(OC)cc2"

        # Using GDSC2 cell line name
        python main_infer.py --model_path /path/to/model \
            --cell_line_name A549 \
            --drug_smiles "CC(=O)NCCC1=CNc2c1cc(OC)cc2"
    """
    if cell_line_h5ad_file is None and cell_line_name is None:
        raise ValueError(
            "Must provide either --cell_line_h5ad_file or --cell_line_name"
        )
    if cell_line_h5ad_file is not None and cell_line_name is not None:
        raise ValueError(
            "Provide only one of --cell_line_h5ad_file or --cell_line_name"
        )

    # Load cell line data as AnnData object
    if cell_line_name is not None:
        print(f"Loading cell line '{cell_line_name}' from GDSC2")
        adata = load_gdsc_cell_line(cell_line_name)
    elif cell_line_h5ad_file is not None:
        print(f"Loading cell line from h5ad file: {cell_line_h5ad_file}")
        adata = sc.read_h5ad(cell_line_h5ad_file)
    else:
        raise ValueError(
            "Either cell_line_name or cell_line_h5ad_file must be provided"
        )

    model = Mammal.from_pretrained(model_path)
    model.eval()
    model.to(device=device)

    tokenizer_op = ModularTokenizerOp.from_pretrained(
        "ibm/biomed.omics.bl.sm.ma-ted-458m"
    )

    print("=" * 80)
    print("Model and tokenizer loaded successfully.")
    print("=" * 80)

    prediction = cell_line_drug_infer(
        model=model,
        tokenizer_op=tokenizer_op,
        adata=adata,
        drug_smiles=drug_smiles,
        device=device,
    )

    print("\n" + "=" * 80)
    print("PREDICTION RESULT:")
    print("=" * 80)
    if cell_line_name:
        print(f"Cell Line: {cell_line_name}")
    if cell_line_h5ad_file:
        print(f"Cell Line: {cell_line_h5ad_file}")
    if drug_name:
        print(f"Drug: {drug_name}")
    print(f"Predicted IC50: {prediction:.6f}")
    print("=" * 80)


def load_gdsc_cell_line(cell_line_id: str) -> ad.AnnData:
    """
    Load a cell line from GDSC2 and return as AnnData object.

    Args:
        cell_line_id: Cell line identifier (e.g., "A549", "FADU")

    Returns:
        AnnData object containing gene expression data
    """
    import pandas as pd
    from tdc.multi_pred import DrugRes

    data = DrugRes(name="GDSC2")
    df = data.get_data()

    cell_line_data = df[df["Cell Line_ID"] == cell_line_id]

    if len(cell_line_data) == 0:
        raise ValueError(f"Cell line '{cell_line_id}' not found in GDSC2")

    print(f"Found {len(cell_line_data)} drug responses for {cell_line_id}")

    expression = cell_line_data.iloc[0]["Cell Line"]
    if not isinstance(expression, np.ndarray):
        expression = np.array(expression)

    gene_symbols = list(data.get_gene_symbols())

    adata = ad.AnnData(
        X=expression.reshape(1, -1),
        var=pd.DataFrame(index=gene_symbols),
        obs=pd.DataFrame({"cell_line_id": [cell_line_id]}, index=[cell_line_id]),
    )

    return adata


def cell_line_drug_infer(
    model,
    tokenizer_op,
    adata: ad.AnnData,
    drug_smiles: str,
    device: str = "cpu",
):
    """
    Predicts IC50 of a given drug over a given cell-line gene expression using MAMMAL a fine-tuned model checkpoint.

    :param model: Pre-loaded MAMMAL model
    :param tokenizer_op: Pre-loaded tokenizer
    :param adata: AnnData object containing gene expression data
    :param drug_smiles: SMILES representation of the drug
    :param device: Device to use for inference
    :return: Prediction value
    """
    encoder_inputs_max_seq_len = 1500
    genes = adata.var_names.tolist()

    if adata.n_obs > 0:
        if issparse(adata.X):
            expressions = np.array(adata.X[0].todense()).flatten()
        else:
            expressions = np.array(adata.X[0]).flatten()
    else:
        raise ValueError("No cells found in AnnData object")

    sample_dict = {
        "genes": genes,
        "expressions": expressions,
        "smiles": drug_smiles,
    }
    sample_dict = CellLineDrugResponseTask.data_preprocessing(
        sample_dict=sample_dict,
        genes_key="genes",
        expressions_key="expressions",
        drug_smiles_key="smiles",
        tokenizer_op=tokenizer_op,
        encoder_input_max_seq_len=encoder_inputs_max_seq_len,
        device=model.device,
    )

    batch_dict = model.forward_encoder_only([sample_dict])

    if SCALARS_PREDICTION_HEAD_LOGITS in batch_dict:
        predictions_full = batch_dict[SCALARS_PREDICTION_HEAD_LOGITS][0]
        prediction = predictions_full[0].squeeze().item()
        return prediction

    print("WARNING: No prediction found")
    return None


if __name__ == "__main__":
    main()

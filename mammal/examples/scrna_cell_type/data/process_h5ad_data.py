import anndata
import click
import numpy as np
import scanpy as sc


@click.command()
@click.option(
    "--input-h5ad-file",
    "-i",
    prompt=True,
    help="name of input H5AD file",
)
@click.option(
    "--output-h5ad-file",
    "-o",
    prompt=True,
    help="name of output H5AD file",
)
@click.option(
    "--min_genes",
    "-m",
    type=click.INT,
    help="minimal number of different genes per cell.  Used for filtering",
    default=200,
)
@click.option(
    "--normalize_total",
    "-n",
    type=click.FLOAT,
    help="Value to normelize the sum or counts to",
    default=1000.0,
)
@click.option(
    "--num_bins",
    "-b",
    type=click.INT,
    help="number of expression bins to use",
    default=11,
)
def main(
    input_h5ad_file: str,
    output_h5ad_file: str,
    min_genes: int = 200,
    normalize_total: float = 1000,
    num_bins: int = 11,
):

    anndata_object = anndata.read_h5ad(input_h5ad_file)
    # process the data - filter out cells with shallow reads, normelize depth and change to log scale of about 0-10 (log_2(1001)~=10)

    sc.pp.filter_cells(anndata_object, min_genes=min_genes)
    sc.pp.normalize_total(adata=anndata_object, target_sum=normalize_total)
    sc.pp.log1p(anndata_object, base=2)

    # split range to bins - more or less 0,2,3,..10
    bins = np.linspace(
        anndata_object.X.data.min(), anndata_object.X.max(), num=num_bins
    )
    print(f"location of bin ends: {bins}")

    # convert the counts to bins
    anndata_object.X.data = np.digitize(anndata_object.X.data, bins)

    # Save result anndata object to disk
    anndata_object.write_h5ad(output_h5ad_file)


if __name__ == "__main__":
    main()

import anndata
import click

from mammal.examples.scrna_cell_type.pl_data_module import preprocess_ann_data


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
    "--min-genes",
    "-m",
    type=click.INT,
    help="minimal number of different genes per cell.  Used for filtering",
    default=200,
)
@click.option(
    "--num-bins",
    "-b",
    type=click.INT,
    help="number of expression bins to use",
    default=10,
)
def main(
    input_h5ad_file: str,
    output_h5ad_file: str,
    min_genes: int = 200,
    num_bins: int = 10,
):

    anndata_object = anndata.read_h5ad(input_h5ad_file)
    # process the data - filter out cells with shallow reads, normelize depth and change to log scale of about 0-10 (log_2(1001)~=10)
    preprocess_ann_data(
        anndata_object=anndata_object,
        min_genes=min_genes,
        num_bins=num_bins,
    )
    # Save result anndata object to disk
    anndata_object.write_h5ad(output_h5ad_file)
    print(f"processed AnnData file saved to {output_h5ad_file}")


if __name__ == "__main__":
    main()

# script to pack zheng68k data downloaded from x10genomics into an AnnData/h5ad file.

# This example follows [Zheng]() for identification of white blood cell types from single cell RNA expression data.


# ## Outline of process
# The finetune process requires the input data to be in scRNA-sec AnnData format (saved as an h5ad file) with cell types
# as labels. If the data is not packed as AnnData, as is the case when downloading from the
# 10xGenomics site as explained below, it need to first be packed into one and saved to the disk.
# cell types, if present, should be stored in the `adata.obs['cell_type']` observation.

# This script assumes that it is run from the data directory,
# which is typically `bmfm-mammal-release/mammal/examples/scrna_cell_type/data`

import os
import subprocess
from pathlib import Path

import anndata
import click
import pandas as pd
from scipy.io import mmread

### Obtaining the source data:
# The main data is available online, for example in the [10xGenomics](https://www.10xgenomics.com/) cite.
# The labels are based on the data in [LINK](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0)
# From this site download the file `fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz` and place it in the data directory.


DEFULT_LABELS_FILE = (
    "zheng17_bulk_lables.txt"  # yes, the original file is named this way.
)
GZIP_FILE_NAME = "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
RAW_H5AD_FILE = "Zheng_68k.h5ad"
RAW_DATA_SUBDIR = Path("filtered_matrices_mex/hg19")


@click.command()
@click.option(
    "--output-h5ad-file",
    "-o",
    default=None,
    help="name of output H5AD file.  default is adding '_preprocessed' to the input file",
)
@click.option(
    "--data-dir",
    help="dirname for the downloaded and constructed data files",
    default=".",
)
@click.option(
    "--labels_file",
    "-l",
    default=DEFULT_LABELS_FILE,
)
@click.option(
    "--labels_key",
    "-k",
    default="cell_type",
    help="key to use for the cell type labels in the AnnData observations.",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="be verbose (default: off)."
)
def main(
    output_h5ad_file: str,
    data_dir: os.PathLike,
    labels_file: os.PathLike,
    labels_key: str,
    verbose: bool = False,
):
    # all work is done in the data dir
    os.chdir(data_dir)
    barcode_file = RAW_DATA_SUBDIR / "barcodes.tsv"
    genes_file = RAW_DATA_SUBDIR / "genes.tsv"
    matrix_file = RAW_DATA_SUBDIR / "matrix.mtx"

    if not RAW_DATA_SUBDIR.exists():
        # check if the file exists
        if not os.path.exists(GZIP_FILE_NAME):
            print(
                f"please download the file {GZIP_FILE_NAME} from https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0 into this data directory and then run this script again from that directory"
            )
            raise FileNotFoundError(
                f"Both the {GZIP_FILE_NAME} and the raw data directory {RAW_DATA_SUBDIR} extracted from it not found under the current directory"
            )
        else:
            if verbose:
                print(f"extracting files from  {GZIP_FILE_NAME}")
            subprocess.run(["tar", "xvzf", GZIP_FILE_NAME], check=True)

    if labels_file is not None:  # if we do not want to add labels
        if not os.path.exists(labels_file):
            if (
                labels_file == DEFULT_LABELS_FILE
            ):  # special case - we can download this file if needed.
                labels_file_url = "https://raw.githubusercontent.com/scverse/scanpy_usage/refs/heads/master/170503_zheng17/data/zheng17_bulk_lables.txt"
                if verbose:
                    print(f"Missing cell-type-labels file {labels_file}")
                    print(f"downloading it from {labels_file_url}")
                subprocess.run(["wget", labels_file_url], check=True)
                if verbose:
                    print("downloaded")
            else:
                raise FileNotFoundError("please supply labels file")
    raw_adata = create_anndata_from_csv(
        barcode_file,
        genes_file,
        matrix_file,
        labels_file=labels_file,
        labels_key=labels_key,
    )

    # Save result anndata object to disk
    raw_adata.write_h5ad(output_h5ad_file)

    if verbose:
        print(f"the raw {output_h5ad_file} is ready")
        print(
            "to use this h5ad file please filter, normalize and bin the counts.  You can use process_h5ad_data.py to do this."
        )

        # This is a standard AnnData file, and can be replaced by your own data.


def create_anndata_from_csv(
    barcode_file,
    genes_file,
    matrix_file,
    labels_file,
    labels_key,
) -> anndata.AnnData:
    """Construct an h5ad file (an anndata object dump) from its components.


    Args:
        raw_h5ad_file (os.PathLike): name of file to save constructed AnnData into
        barcode_file (os.PathLike): this file holds the mapping from the sample index to the cell identifier
        genes_file (os.PathLike): Mapping from feature index to gene name
        matrix_file (os.PathLike): The actual data, is (sparse) matrix form.
        labels_file (os.PathLike, optional): File containing the cell types for each file.
        labels_key (str): name of observation to place labels under in the AnnData object.
        verbose (bool, optional): verbose output. Defaults to False.

    Returns:
        anndata.AnnData: the generated anndata object
    """
    mmx = mmread(matrix_file)

    #### Create an AnnData object wrapping the read data

    # Notice that this code transposes the data to the correct direction

    anndata_object = anndata.AnnData(X=mmx.transpose().tocsr())

    # Cell identifiers
    observation_names = pd.read_csv(barcode_file, header=None, sep="\t")
    # names of genes
    genes = pd.read_csv(genes_file, header=None, sep="\t")

    # use the gene names as variable names in the AnnData object
    anndata_object.var_names = genes[1]

    # use the cell barcodes as names for the samples
    anndata_object.obs_names = observation_names[0]

    if labels_file is not None:
        # cell types (this is actualy just one column)
        cell_type_labels = pd.read_csv(labels_file, header=None, sep="\t")
        # use cell types as labels for the samples
        anndata_object.obs[labels_key] = cell_type_labels.squeeze().to_numpy()

    return anndata_object


if __name__ == "__main__":
    main()

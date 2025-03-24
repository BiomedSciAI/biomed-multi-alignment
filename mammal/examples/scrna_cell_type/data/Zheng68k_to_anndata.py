# # script to pack zheng68k data from x10genomics into an AnnData (h5ad) file.

# This example follows [Zheng]() for identification of white blood cell types from single cell RNA expression data.


# ## Outline of process
# The finetune process requires the input data to be in AnnData (h5ad) format with the cell types
# as labels. If the data is not packed as AnnData, as is the case when downloading from the
# 10xGenomics site as explained below, it need to first be packed into one and saved to the disk.

# Once the data is in AnnData format, it needs to be canonized via data and variable name transformations.
# After the transformations the filtered and canonized data may be saved again in AnnData format or used directly.  This data can be used by the training process.

# In the training process, each sample is transformed into a GeneFormer like ordered list of genes, which is done by sorting all the genes by binned expression level and within each bin alphabetically by the gene name (as it is used in the tokenizer).
# This is done inside the task.process_model_output method, together with input preparations and tokenizations.


# ## preprocessing data transformations

# * Filter out cell with less then 200 active genes

#         scanpy.pp.filter_cells(anndata_object,min_genes=200)


# * Normelize the sum of counts for each cell to a constant (1000)

#         scanpy.pp.normalize_total(anndata_object,1000.)


# * Move to log space (note,the data prior to this step will be in the range 0-10 so scanpy may issue a warning that the data has allready beed log-scaled)

#         scanpy.pp.log1p(anndata_object,base=2)


# * split the full range of values into bins and digitize the values

#         bins=np.linspace(anndata_object.X.data.min(), anndata_object.X.max(),num=10)
#         anndata_object.X.data=np.digitize(anndata_object.X.data, bins)

#     Note that this was done over all the data, but that is not likely to cause any bleeding from the test sets


# This notebook assumes that it is run from the `bmfm-mammal-release/mammal/examples/scrna_cell_type` directory ("the current directory").

import os
import subprocess
from pathlib import Path

import anndata
import click
import pandas as pd
from scipy.io import mmread

# this script is meant to be run from the `data` directory under the scrna_cell_type example directory.

### Obtaining the source data:
# The main data is available online, for example in the [10xGenomics](https://www.10xgenomics.com/) cite.  The labels are based on the data in [LINK](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0)
# From this site download the file `fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz` and place it in this directory.


@click.command()
@click.option(
    "--input-h5ad-file",
    "-i",
    default=None,
    help="name of input H5AD file;   if not supplied the file will be constructed from Zheng86k data",
)
@click.option(
    "--output-h5ad-file",
    "-o",
    default=None,
    help="name of output H5AD file.  default is adding '_preprocessed' to the input file",
)
@click.option(
    "--raw-data-dir",
    help="dirname with barcodes.tsv, genes.tsv, and matrix.mtx files",
    default="filtered_matrices_mex/hg19",
)
@click.option(
    "--min-genes",
    "-m",
    type=click.INT,
    help="minimal number of different genes per cell.  Used for filtering in pre-processing",
    default=200,
)
@click.option(
    "--normalize-total",
    "-n",
    type=click.FLOAT,
    help="Value for normalizing the sum of counts in preprocessing",
    default=1000.0,
)
@click.option(
    "--num-bins",
    "-b",
    type=click.INT,
    help="number of expression bins to use in pre-processing",
    default=10,
)
@click.option(
    "--pre-process",
    "-p",
    is_flag=True,
    default=True,
    help="do not pre-process raw AnnData (default: pre-process).",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="be verbose (default: off)."
)
def main(
    input_h5ad_file: str,
    output_h5ad_file: str,
    raw_data_dir: os.PathLike,
    min_genes: int = 200,
    normalize_total: float = 1000,
    num_bins: int = 10,
    pre_process: bool = True,
    verbose: bool = False,
):
    # Unless a raw h5ad file is given, one needs to be constructed from it's parts
    if input_h5ad_file is None:
        raw_h5ad_file = "Zheng_68k.h5ad"
        raw_data_dir = Path(raw_data_dir)
        barcode_file = raw_data_dir / "barcodes.tsv"
        genes_file = raw_data_dir / "genes.tsv"
        matrix_file = raw_data_dir / "matrix.mtx"
        labels_file = "zheng17_bulk_lables.txt"

        if not raw_data_dir.exists():
            gzip_file_name = "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"

            # check if the file exists
            if not os.path.exists(gzip_file_name):
                print(
                    f"please download the file {gzip_file_name} from https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0 into this data directory and then run this script again from that directory"
                )
                raise FileNotFoundError(
                    f"Both the {gzip_file_name} and the raw data directory {raw_data_dir} extracted from it not found under the current directory"
                )
            else:
                if verbose:
                    print(f"extracting files from  {gzip_file_name}")
                subprocess.run(["tar", "xvzf", gzip_file_name])

        if labels_file is not None:  # if we do not want to add lables
            if not os.path.exists(labels_file):
                if labels_file == "zheng17_bulk_lables.txt":
                    labels_file_url = "https://raw.githubusercontent.com/scverse/scanpy_usage/refs/heads/master/170503_zheng17/data/zheng17_bulk_lables.txt"
                    if verbose:
                        print(f"Missing cell-type-labels file {labels_file}")
                        print(f"downloading it from {labels_file_url}")
                    subprocess.run(["wget", labels_file_url])
                    if verbose:
                        print("downloaded")
                else:
                    raise FileNotFoundError("please supply labels file")
        input_h5ad_file = create_anndata_from_csv(
            raw_h5ad_file,
            barcode_file,
            genes_file,
            matrix_file,
            labels_file=labels_file,
            verbose=verbose,
        )

        # This is a standard AnnData file, and can be replaced by your own data.

    if pre_process:
        # if we don't have an output file name, create it by adding "_preprocessed" between the input file name and the file extension
        preprocessing_kwargs = {
            "--min-genes": min_genes,
            "--normalize-total": normalize_total,
            "--num-bins": num_bins,
        }

        pre_process_anndata_file(
            input_h5ad_file,
            output_h5ad_file=output_h5ad_file,
            preprocessing_kwargs=preprocessing_kwargs,
            verbose=verbose,
        )


def create_anndata_from_csv(
    raw_h5ad_file,
    barcode_file,
    genes_file,
    matrix_file,
    labels_file="zheng17_bulk_lables.txt",
    verbose=False,
) -> os.PathLike:
    """Construct an h5ad file (an anndata object dump) from its components.


    Args:
        raw_h5ad_file (os.PathLike): name of file to save constructed AnnData into
        barcode_file (os.PathLike): this file holds the mapping from the sample index to the cell identifier
        genes_file (os.PathLike): Mapping from feature index to gene name
        matrix_file (os.PathLike): The actual data, is (sparse) matrix form.
        labels_file (os.PathLike, optional): File containing the cell types for each file. Defaults to "zheng17_bulk_lables.txt".
        verbose (bool, optional): verbose output. Defaults to False.

    Returns:
        os.PathLike: name of h5ad file
    """

    #  You should now have a directory called `filtered_matrices_mex/hg19`
    # with the following files
    #   1. barcodes.tsv
    #   2. genes.tsv
    #   3. matrix.mtx

    # ## Pack the data into an AnnData file
    #### Read the scRNA matrix from a file
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
        cell_type_lables = pd.read_csv(labels_file, header=None, sep="\t")
        # use cell types as labels for the samples
        anndata_object.obs["celltype"] = cell_type_lables.squeeze().to_numpy()
        # Save result anndata object to disk
    anndata_object.write_h5ad(raw_h5ad_file)

    if verbose:
        print(f"the raw {raw_h5ad_file} is ready")
    return raw_h5ad_file


def pre_process_anndata_file(
    input_h5ad_file, output_h5ad_file, preprocessing_kwargs={}, verbose=False
):
    if output_h5ad_file is None:
        file_name, file_extension = os.path.splitext(input_h5ad_file)
        output_h5ad_file = file_name + "_preprocessed" + file_extension

    if verbose:
        print(f"Preprocessing AnnData file {input_h5ad_file}")
    if output_h5ad_file is None:
        file_name, file_extension = os.path.splitext(input_h5ad_file)
        output_h5ad_file = file_name + "_preprocessed" + file_extension

    # The code requires a simple preprocessing stage to get it to a standard form (counts).  [../process_h5ad_data.py] is a script that can be used to run the preprocessing stage.

    if verbose:
        print(
            "you can see preprocessing commandline options with 'python process_h5ad_data.py --help' if you wish to see the options available"
        )

    anndata_preprocessing_arguments = [
        "python",
        "process_h5ad_data.py",
        "--input-h5ad-file",
        input_h5ad_file,
        "--output-h5ad-file",
        output_h5ad_file,
        "",
    ]
    for key, value in preprocessing_kwargs.items():
        anndata_preprocessing_arguments.append([f"--{key}", value])
    subprocess.run(anndata_preprocessing_arguments, check=True)

    # !ls -sh1 *.h5ad
    # ```
    # 917616 Zheng_68k.h5ad
    # 886288 Zheng_68k_preprocessed.h5ad
    # ```

    if verbose:
        print(
            f"The AnnData file '{output_h5ad_file}' contains the filtered and preprocessed data in the required format"
        )


if __name__ == "__main__":
    main()

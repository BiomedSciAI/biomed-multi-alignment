# # Notebook to pack zheng68k data from x10genomics into an AnnData (h5ad) file.

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

# There is a commented example of this process at the end of this notebook.

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
# # check the current directory.

# this is just a fancy pwd, replace with !pwd if you need
# !print -D $PWD
# expecting the answer to be something like
# `~/git/biomed-multi-alignment/mammal/examples/scrna_cell_type/data`
#
#  Notice that the `biomed-multi-alignment` will probably be placed in a different location on your system.


import os
import subprocess
from pathlib import Path

import anndata
import pandas as pd
from scipy.io import mmread

# this script is meant to be run from the `data` directory under the scrna_cell_type example directory.

### Obtaining the source data:
# The main data is available online, for example in the [10xGenomics](https://www.10xgenomics.com/) cite.  The labels are based on the data in [LINK](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0)
# From this site download the file `fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz` and place it in this directory.

#### Unzip the file.
# !tar -xzvf fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz

raw_data_dir = Path("filtered_matrices_mex/hg19")
barcode_file = raw_data_dir / "barcodes.tsv"
genes_file = raw_data_dir / "genes.tsv"
matrix_file = raw_data_dir / "matrix.mtx"

if not raw_data_dir.exists():
    gzip_file_name = "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"

    # check if the file exists
    if not os.path.exists(gzip_file_name):
        print(
            f"please download the file {gzip_file_name} from https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0 into this data directory and then run this script again from that directory"
        )
        raise FileNotFoundError(
            f"Both the raw data directory {raw_data_dir} and {gzip_file_name} not found under the current directory"
        )
    else:
        print(f"extracting files from  {gzip_file_name}")
        subprocess.run(["tar", "xvzf", gzip_file_name])
        # print(
        #     f"please unzip {gzip_file_name} in the data directory and then run this script again from that directory"
        # )
        # raise FileNotFoundError(
        #     f"the raw data directory {raw_data_dir} not found under the current directory"
        # )

#### Download the labels file from a git repository in https://github.com/scverse/scanpy_usage
# !wget https://raw.githubusercontent.com/scverse/scanpy_usage/refs/heads/master/170503_zheng17/data/zheng17_bulk_lables.txt

labels_file = "zheng17_bulk_lables.txt"
if not os.path.exists(labels_file):
    labels_file_url = "https://raw.githubusercontent.com/scverse/scanpy_usage/refs/heads/master/170503_zheng17/data/zheng17_bulk_lables.txt"
    print(f"Missing cell-type-labels file {labels_file}")
    print(f"downloading it from {labels_file_url}")
    subprocess.run(["wget", labels_file_url])
    print("downloaded")

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
assert anndata_object.X.shape == (68579, 32738), "AnnData data is in an unexpected size"

# Cell identifiers
barcodes = pd.read_csv(barcode_file, header=None, sep="\t")
# names of genes
genes = pd.read_csv(genes_file, header=None, sep="\t")

# cell types (this is actualy just one column)
cell_type_lables = pd.read_csv("zheng17_bulk_lables.txt", header=None, sep="\t")

# use the gene names as variable names in the AnnData object
anndata_object.var_names = genes[1]

# use the cell barcodes as names for the samples
anndata_object.obs_names = barcodes[0]


# use cell types as labels for the samples
anndata_object.obs["celltype"] = cell_type_lables.squeeze().to_numpy()
# Save result anndata object to disk
raw_anndata_file = "Zheng_68k.h5ad"
anndata_object.write_h5ad(raw_anndata_file)

print(f"the raw {raw_anndata_file} is ready")
# This is a standard AnnData file, and can be replaced by your own data.
print("Preprocessing calls' reads")

# The code requires a simple preprocessing stage to get it to a standard form.  [../process_h5ad_data.py] is a script that can be used to run the preprocessing stage.


# The parameters of the processing can be controlled from the command line.  Use

# ! python process_h5ad_data.py --help

# for a list of available parameters

# This script can be used to filter and process the AnnData (the "!" indicates to the notebook to run this as a shell command, so remove it for commandline use).
# The output of this is stored in `Zheng_68k_filtered.h5ad`, and us used by the config to run the model


# ! python process_h5ad_data.py --input-h5ad-file Zheng_68k.h5ad --output-h5ad-file Zheng_68k_filtered.h5ad
# The annData file should ready in the data directory.
filtered_anndata_file = "Zheng_68k_filtered.h5ad"
if not os.path.exists(filtered_anndata_file):
    print("need to please preprocess data using")
    print(
        f"python process_h5ad_data.py --input-h5ad-file {raw_anndata_file} --output-h5ad-file {filtered_anndata_file}"
    )
    print(
        "you can see preprocessing commandline options with 'python process_h5ad_data.py --help' if you wish to see the options available"
    )

    subprocess.run(
        [
            "python",
            "process_h5ad_data.py",
            "--input-h5ad-file",
            raw_anndata_file,
            "--output-h5ad-file",
            filtered_anndata_file,
        ]
    )

# !ls -sh1 --color=never *.h5ad
# ```
# 917616 Zheng_68k.h5ad
# 886288 Zheng_68k_filtered.h5ad
# ```
print(
    f"The AnnData file '{filtered_anndata_file}' contains the filtered and preprocessed data in the required format"
)

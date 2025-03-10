# scRNA based cell type Annotation Prediction MAMMAL
This directory contains code to fine-tune a MAMMAL model, as is available on [hugging face](https://huggingface.co/ibm-research/biomed.omics.bl.sm.ma-ted-458m)
and to predict the cell type using the fine-tuned model

##  Description
### Input for fine-tune:
The required input for the fine-tuning is an [AnnData](https://anndata.readthedocs.io/en/stable/) file with scRNA samples where:
* Each single cell reading is a sample
*  The samples have genes for variables
*  The values stored in the data part (`X` section) are counts for the gene in the sample
*  The cell type of the sample is stored in the `obs['celltype']` observation for the sample.

### Input for prediction:
Similar to the input for fine-tuning, but the cell-types observations are not needed and will be ignored if present.

## Data and data preparation
Input needs to be saved to an AnnData file (ending with `.h5ad`) with all the names of genes and cell types consistent with the format in the tokenizer

The [data/process_h5ad_data.py](data/process_h5ad_data.py) script runs the data preprocessing needed to convert the raw counts to the expected input.  This process consists of
 1. filtering the cells to remove cells with less then 200 different samples
 2. normalizing the total counts for all the reads of the cell to 1000
 3. passing the counts through `log(value+1)` which shifts the counts to the range of zero to nine.
 4. Binning (or Digitizing) the values to the

 See [preprocess_ann_data in pl_data_module.py](pl_data_module.py#L225) for an implementation and the details.


## Example data for this demo

To help understand the workings of the code, one can use the scRAN and cell-type annotation from the Zheng68k dataset.  This data needs to be downloaded, packed into AnnData/ an H5AD file and preprocessed to be useable.
The [data](data) directory contains a script that does most of this work, but requires the user to obtain the data (see below).  The script can be used as a reference for packing other scRNA data into the expected format, and contains explanation on the steps needed.

**See [Steps needed to run the demo on the Zheng68k data](#steps-needed-to-run-the-demo-on-the-zheng68k-data) below for a detailed walkthrough.**


An example for packing data into an AnnData file, the
[Zheng68k_to_anndata.py](data/Zheng68k_to_anndata.py) script, can be found in the data subdirectory.
To use this script you will need to download fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz from [Fresh 68k PBMCs (Donor A) dataset](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0) from the [10xgenomics website](https://www.10xgenomics.com) (may require filling a form) or some other source.
place this file in the data directory, cd into it and run  `python Zheng68k_to_anndata.py`.


The script includes instructions and code for downloading, processing and packing the [Zheng68k](TODO:link) dataset.  It also contains information about preprocessing the AnnData file to a the expected input format and information about that format.

## GeneFormer inspired ordered gene encoding string

MAMMAL is a transformer who's input is a string of tokens.  scRNA data is typically collected as pairs of **(gene name,expression level)**.  We chose to perform this transformation by
binning the expressions into (typically) 10 bins based on expression, and then sort them by bin number (from largest expression down). All the genes inside each bin are considered to be of equivalent expression levels. To create a consistent, MLM learnable output, we sort the genes within each bin lexicographically by gene name.  This can be replaced by any other arbitrary and constant gene order, as long as the ordering scheme is used in training and test.

After this double sorting, the expression level (or bin number) are ignored, and the expression profile is represented by the list of gene names in this order.  This list is then used as the representation of the cell's scRNA, as inputted into the model.  For performance reasons, the string is truncated to a fixed size (500-2000 typically, there is a parameter in the config file to change this).

This stage is done while reading the data, and is not part of the preprocessing.

    def convert_to_double_sorted_geneformer_sequence(anndata_object):

        # the genes are sorted by expression bin (descending) and within the bin by the gene names.
        return [a[1] for a in sorted(zip(-anndata_object.X.data,anndata_object.var_names.to_numpy()[anndata_object.X.indices]))]


### Example of the double sorting and ordering:
Assuming that the data was preprocessed and is now in standard gene names and bin numbers pairs.
For this example, we will consider a sample with four genes, each paired with the corresponding bin number.
As explained above, the genes are first sorted by the bin number (from large to small) and then within each bin the genes are sorted lexicographically by name.  Finally, the bin numbers are dripped and the genes are presented to the model as an ordered sequence.


||stage| **name** | **bin**  |***\\***|    |  **name** | **bin**   |***\\***|  **name** | **bin**    |***\\***|  **name** |**bin**    |
|:-:|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|| | ||| | | ||
1| input data | ABCD | 4 |***\\***|| ZZTP | **6**  | ***\\***| BRCA | 6 |***\\***|MINI | 6 |
2|sorted by bin| ZZTP | 6 | ***\\***||   BRCA | **6** | ***\\***|MINI | 6  |***\\***|ABCD | 4
3|sorted by name| **B**RCA | 6 | ***\\***|| **M**INI | **6**  | ***\\***| **Z**ZTP | 6 |***\\***|**A**BCD | 4
4|bins removed|  BRCA | | || MINI |   | | ZZTP |  ||ABCD |



1. The input (preprocessed) data is ***[(ABCD,4),( ZZTP,6),(BRCA,6),( MINI,6)]***
2. The data is first sorted by the bin (in reverse id so it will be form the larges to the smallest)
3. Within each bin group the data is sorted by the name
4. Finally the bins are removed and the sequence of genes is used.

The result is the string
***"ABCD,ZZTP,BRCA,MINI"*** which is used as the input for the LLM


## Running fine-tune
The package's [main readme](../../../README.md) contains instructions of running fine-tune.  As explained there, you may need to modify the [config.yaml](config.yaml) file, mainly in the `task` section.  The config is set up under the assumption that the data file created in the [Zheng68k_data_prep.ipynb](data/Zheng68k_data_prep.ipynb) notebook, can be read from the `data` subdirectory of the example

## Steps needed to run the demo on the Zheng68k data:

1. Install the  `biomed-multi-alignment` package with the examples:

    ```cd biomed-multi-alignment & pip install -e '.[examples]'```

2.  From the [10xgenomics website](https://www.10xgenomics.com), Download

    `fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz`

    which can be found at [Fresh 68k PBMCs (Donor A) dataset](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0)
    under **"Output and supplemental files -> Gene / cell matrix (filtered)"**


3. Place file in the data directory

    **biomed-multi-alignment/mammal/examples/scrna_cell_type/data**

4. Go into the data directory and run the data preparation script

    ```python Zheng68k_to_anndata.py```

    This will produce a file called `Zheng_68k_preprocessed.h5ad` in the data directory.

    Use `python Zheng68k_to_anndata.py --help` to see the possible command line options which control the pre-processing process.

5. Edit [biomed-multi-alignment/mammal/examples/scrna_cell_type/config.yaml](biomed-multi-alignment/mammal/examples/scrna_cell_type/config.yaml) if needed to change the parameters of the training.  The default setup should work fine.

6. return to the top directory **`biomed-multi-alignment`** of the project
    and run the main finetune program:

    ```% python ./mammal/main_finetune.py "--config-name=config.yaml" "--config-path=examples/scrna_cell_type"```

When this is done your model should be fine-tuned for the task

# scRNA based cell type Annotation Prediction MAMMAL
This directory contains code to fine-tune a MAMMAL model, as is available on [hugging face](https://huggingface.co/ibm-research/biomed.omics.bl.sm.ma-ted-458m)
and to predict the cell type using the fine-tuned model.

Also included are scripts to build an AnnData from the Zheng68k data (see below) that are used for the fine tune, and a script to prepare the AnnData file for Mammal.

##  Description
### Input structure:
The required input is an scRNA-seq [AnnData](https://anndata.readthedocs.io/en/stable/) structure saved in an h5ad (or similar) file.
AnnData (for **Annotated Data**) is specifically designed for matrix-like data with meta data on both the samples and the variables.
You can find explanations on its structure of and specifically AnnData for scRNA-seq data in [this AnnData tutorial](https://anndata.readthedocs.io/en/stable/tutorials/notebooks/getting-started.html).

#### Standard AnnData structure for scRNA-seq:
 In scRNA-seq AnnData, each row corresponds to a cell with a barcode, and each column corresponds to a gene with a gene id.  In addition to this, it may contain meta-data regarding the genes (like alternative gene symbols) or the observations (such as cell-type). `adata.X` contains the main data in a sparse cell-id by gene cell  data matrix which is typically the counts for the gene in the observation.

This translates into the following:
*  Each data observation represent the gene expression of a single cell.
*  The observations have a barcode id stored in `adata.obs_names`
*  The variable names for each observation are the gene symbols, stored in `anndata.var_names`
*  The values stored in the data part (`adata.X` section) are counts for the relevant gene for the observation, and


## Data and data preparation
Input needs to be packed to an AnnData file (ending with `.h5ad`) as described above.

### Additional AnnData requirements specific to this demo:
The standard AnnData scRNA-seq data format requires some small changes to fit the code.
The code assumes that *all* the variables are gene counts, but AnnData allows other variables, so

*  Only counts of genes are in the data vector - other variables must first be removed.

The code uses the `"cell_type"` observation as the sample's class, so at least for training:

*  The cell type of the observations are stored in the `adata.obs['cell_type']` observation.

The key can be controlled via the `task.data_module_kwargs.label_name` parameter of the [config.yaml](config.yaml) file.

Note that all the names of genes and cell types need to be consistent with the format/naming schema in the tokenizer.

### Filtration of the cells prior to training:
As is common when working with such data, the cells are filtered to remove partial and suspicious reads.  We used two filters to achieve this - removing cells with a small number of different RNAs in the read, and removing cells with very shallow reads.  See [Filtration and processing](#filtering-and-processing-reference-script) section below for our reference implementation and the parameters we used.  Depending on your data, you may want to modify this process.

### Digitizing the counts
As is explained in the [encoding section below](#geneformer-inspired-ordered-gene-encoding-string), the encoding of the cell reads for the model
requires that the counts for each gene will be **replaced** with a digitized version of the value, which corresponds to the bin the value falls into.
The bins are assumed to be numbered `0,1,...,n-1' where smaller bin number represents a bigger count.

The code assumes that the data has been filtered as needs and then digitized and saved into an h5ad file with the bin numbers *replacing* the counts in the `data.X` matrix.

### Filtering and processing reference script
The [data/process_h5ad_data.py](data/process_h5ad_data.py) script runs the data preprocessing as described above.  This process consists of
 1. filtering the cells to remove cells with less then 200 different samples
 2. normalizing the total counts for all the reads of the cell to 1000
 3. passing the counts through `log(value+1)` which shifts the counts to the range of zero to nine.
 4. Binning (or Digitizing) the transformed counts into n bins numbered 0,1,..,n-1
    Binning is done uniformly on the full range of the transformed counts.
 See [preprocess_ann_data in pl_data_module.py](pl_data_module.py#L225) for an implementation and the details.

 Note that the parameters of the preprocessing can be changed via command line arguments.

This script is used to convert from an `input.h5ad` file to a `processed.h5ad` file, with the cells that passed the filters and the binned data.

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

MAMMAL is a transformer who's input is a string of tokens.  scRNA data is typically collected as pairs of **(gene name,expression level)**.  We chose to perform this transformation from genes and counts to tokens by
binning the expressions into (typically) 10 bins based on expression, and then sort them by bin number (from largest expression down). All the genes inside each bin are considered to be of equivalent expression levels. To create a consistent, MLM learnable output, we sort the genes within each bin lexicographically by gene name.  This can be replaced by any other arbitrary and constant gene order, as long as the ordering scheme is used in training and test.

After this double sorting, the expression level (or bin number) are ignored, and the expression profile is represented by the list of gene names in this order.  This list is then used as the representation of the cell's scRNA, as inputted into the model.  For performance reasons, the string is truncated to a fixed size (500-2000 typically, there is a parameter in the config file to change this).

This process was used when pre-training the base model on scRNA data, and consistent processing should help the fine-tuning process.

This sorting/tokenizing stage is done while reading the data, and is not part of the preprocessing.

Here is a reference code in python for performing this double sorting process, to help understand what exactly and how this is done in the code:

    def convert_to_double_sorted_geneformer_sequence(anndata_object):

        # the genes are sorted by expression bin (descending) and within the bin by the gene names.
        return [a[1] for a in sorted(zip(-anndata_object.X.data,anndata_object.var_names.to_numpy()[anndata_object.X.indices]))]


### Example of the double sorting and ordering:
Assuming that the data was  [preprocessed](#data-preprocessing) and is now in standard gene names and bin numbers pairs.
For this example, we will consider a sample with four genes, each paired with the corresponding bin number.
As explained above, the genes are first sorted by the bin number (from large to small) and then within each bin the genes are sorted lexicographically by name.  Finally, the bin numbers are dripped and the genes are presented to the model as an ordered sequence.


||stage| **name** | **bin**  |***\\***|    |  **name** | **bin**   |***\\***|  **name** | **bin**    |***\\***|  **name** |**bin**    |
|:-:|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
1| input data | ABCD | 4 |***\\***|| ZZTP | 6  | ***\\***| BRCA | 6 |***\\***|MINI | 6 |
2|sorted by bin| ZZTP | **6** | ***\\***||   BRCA | **6** | ***\\***|MINI | **6**  |***\\***|ABCD | **4**
3|sorted by name| **B**RCA | 6 | ***\\***|| **M**INI | 6  | ***\\***| **Z**ZTP | 6 |***\\***|**A**BCD | 4
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

In this example we will fine-tune on cell type by doing the following:

1. **Install** the  `biomed-multi-alignment` package with the examples:

    ```cd biomed-multi-alignment & pip install -e '.[examples]'```

### download example data

2.  From the [10xgenomics website](https://www.10xgenomics.com), **Download**

    `fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz`

    which can be found at [Fresh 68k PBMCs (Donor A) dataset](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0)
    under **"Output and supplemental files -> Gene / cell matrix (filtered)"**

### pack Zheng data into AnnData file
3. **Place** file in the data directory

    **biomed-multi-alignment/mammal/examples/scrna_cell_type/data**

4. Go into the data directory and run the **data preparation** script

    ```python Zheng68k_to_anndata.py```

    This script downloads the label and builds an AnnData file from the different data files, and finally it saves the file `Zheng_68k.h5ad` which is a standard scRAN-sec AnnData file.

    Use `python Zheng68k_to_anndata.py --help` to see the possible command line options which control the pre-processing process.

### from standard scRNA-sec AnnData to MAMMAL

5.  run the `data/process_h5ad_data.py` script on the standard AnnData file.
    **Note: this is done automatically as part of the `Zheng68k_to_anndata.py` script, so it does not need to be done manually if this data is used.

    ```python ./process_h5ad_data.py -i Zheng_68k.h5ad -o Zheng_68k_preprocessed.h5ad```

    This will produce a file called `Zheng_68k_preprocessed.h5ad` in the data directory.

    Use `python  ./process_h5ad_data.py --help` to see the possible command line options which control the pre-processing process.


### run the file-tune process

6. Edit [biomed-multi-alignment/mammal/examples/scrna_cell_type/config.yaml](biomed-multi-alignment/mammal/examples/scrna_cell_type/config.yaml) if needed to change the parameters of the training.  The default setup should work fine.

7. return to the top directory **`biomed-multi-alignment`** of the project
    and run the main finetune program:

    ```% python ./mammal/main_finetune.py "--config-name=config.yaml" "--config-path=examples/scrna_cell_type"```

When this is done your model should be fine-tuned for the task


## Inference
Once a fine-tuned model is ready, inference can be done using the `scRNA_infer.py` script:

```python scRNA_infer.py --model-path [model path] --h5ad-file-path [data file] -s [sample id]```

This will print the result of the inference, including cell-type name and token id and the score (both normalized and unnormalized).

## Trouble Shooting
The inference script can be used to identify problems with the data.  Increase verbose level by specifying `-v`, `-vv` or even `-vvv` command line options.  This will print additional intermediate objects like the input data and the prompts.
You may also use `--test-h5ad-file` to run additional diagnostics on the h5ad file and the specific requirements needed to run MAMMAL on it.

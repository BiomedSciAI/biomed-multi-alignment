# scRNA based cell type Annotation Prediction MAMMAL
This directory contains code to fine-tune a MAMMAL model, as is available on [hugging face](https://huggingface.co/ibm-research/biomed.omics.bl.sm.ma-ted-458m)
and to predict the cell type using the fine-tuned model


    Step one, get the data: run the notebook x (I would prefer a script ).
    Step two, modify the data set
    Step three, run this example by..



##  Description
### Input for fine-tune:
The required input for the fine-tuning is an AnnData file with scRNA samples:
* Each single cell reading is a sample, with:
    *  genes for vars
    *  counts for each gene as values
    *  cell type in the `obs['celltype']` observation for the sample.

### Input for prediction:  TODO:??
Similar to the input for fine-tuning, but the cell-types observations are not needed and will be ignored if present.

## Data and data preparation
Input needs to be saved to an AnnData file (ending with `.h5ad`) with all the names of genes and cell types consistent with the format in the tokenizer

The [data/process_h5ad_data.py](data/process_h5ad_data.py) script runs the data preprocessing needed to convert the raw counts to the expected input.  This process consists of
 1. filtering the cells to remove cells with less then 200 different samples
 2. normalizing the total counts for all the reads of the cell to 1000
 3. passing the counts through `log(value+1)` which shifts the counts to the range of zero to nine.
 4. Binning (or Digitizing) the values to the

 See [preprocess_ann_data in pl_data_module.py](pl_data_module.py#L225) for an implementation and the details.

An example for packing data into an AnnData file, the
[Zheng68k_data_prep.ipynb](data/Zheng68k_data_prep.ipynb) notebook, can be found in the data subdirectory.  It includes instructions and code for downloading, processing and packing the [Zheng68k](TODO:link) dataset.  It also contains information about preprocessing the AnnData file to a the expected input format and information about that format.

## GeneFormer inspired ordered gene encoding string

Mammal is a transformer who's input is a string of tokens.  scRNA data is typically collected as pairs of **(gene name,expression level)**.  We chose to perform this transformation by
binning the expressions into (typically) 10 bins based on expression, and then sort them by bin number (from largest expression down). All the genes inside each bin are considered to be of equivalent expression levels. To create a consistent, MLM learnable output, we sort the genes within each bin lexicographically by gene name.  This can be replaced by any other arbitrary and constant gene order, as long as the ordering scheme is used in training and test.

After this double sorting, the expression level (or bin number) are ignored, and the expression profile is represented by the list of gene names in this order.  This list is then used as the representation of the cell's scRNA, as inputted into the model.  For performance reasons, the string is truncated to a fixed size (500-2000 typically, there is a parameter in the config file to change this).

This stage is done while reading the data, and is not part of the preprocessing.

    def convert_to_double_sorted_geneformer_sequence(anndata_object):

        # the genes are sorted by expression bin (descending) and within the bin by the gene names.
        return [a[1] for a in sorted(zip(-anndata_object.X.data,anndata_object.var_names.to_numpy()[anndata_object.X.indices]))]




## Running fine-tune
The package's [main readme](../../../README.md) contains instructions of running fine-tune.  As explained there, you may need to modify the [config.yaml](config.yaml) file, mainly in the `task` section.  The config is set up under the assumption that the data file created in the [Zheng68k_data_prep.ipynb](data/Zheng68k_data_prep.ipynb) notebook, can be read from the `data` of the example

Running is performed with the command:

   ```% python ./mammal/main_finetune.py "--config-name=config.yaml" "--config-path=examples/scrna_cell_type"```

run from the top level directory of `biomed-multi-alignment`

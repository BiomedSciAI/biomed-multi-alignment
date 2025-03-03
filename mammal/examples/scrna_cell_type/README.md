# scRNA based cell type classification using MAMMAL
This directory contains code to finetune a MAMMAL model, as is available on [hugging face](https://huggingface.co/ibm-research/biomed.omics.bl.sm.ma-ted-458m)
 and evaluate




#  TODO: write readme with information on the task and the prompt building


# scRNA based Cell Type Annotation Prediction

##  Description
### Input for fine-tune:
AnnData file with scRNA samples: genes for vars, counts for each gene as values and cell type in the `obs['celltype']` observation for the sample.

### Input for prediction:  TODO:??

 An AnnData file as above but without the cell types
OUTPUT: TODO:??

## Data and data preparation
Input needs to be saved to an AnnData file (ending with `.h5ad`) with all the names of genes and cell types consistent with the format in the tokenizer

An example for packing data into an AnnData file,
[Zheng68k_data_prep.ipynb](data/Zheng68k_data_prep.ipynb), notebook, can be found in the data subdirectory.  It includes instructions and code for downloading, processing and packing the [Zheng68k](TODO:link) dataset.  It also contains information about preprocessing the AnnData file to a the expected input format and information about that format.

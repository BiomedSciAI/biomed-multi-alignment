# Data for Cell Type identification from scRNA data
This example follows [Zheng]() for identification of white blood cell types from single cell RNA expression data.

data is assumed to be in anndata format with the cell types asa labels.

<!-- ## Obtaining the raw data:
The main data is availble online, for example in the [10x]() cite.  The lables are based on the data in[LINK]() -->



## Packing the data into an h5ad (anndata) file
If the original data comes in a several files
* Cell by gene expression count matrix
* Cell identifires to row mapping
* Gene identifires to column mapping
* Cell type by cell identification maping.

it needs to be loaded into an anndata and save to this directory

## data transformations

* Filter out cell with less then 200 active genes

        scanpy.pp.filter_cells(anndata_object,min_genes=200)



* Normelize the sum of counts for each cell to a constant (1000)

        scanpy.pp.normalize_total(anndata_object,1000.)


* Move to log space (note,the data prior to this step will be in the range 0-10 so scanpy may issue a warning that the data has allready beed log-scaled)

        scanpy.pp.log1p(anndata_object,base=2)


* split the full range of values into bins and digitize the values

        bins=np.linspace(anndata_object.X.data.min(), anndata_object.X.max(),num=10)
        anndata_object.X.data=np.digitize(anndata_object.X.data, bins)

    Note that this was done over all the data, but that is not likely to cause any bleeding from the test sets

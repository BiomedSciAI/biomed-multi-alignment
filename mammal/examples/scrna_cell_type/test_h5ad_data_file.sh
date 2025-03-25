#!/bin/bash


# Enable error handling
set -e


# Check if an argument is provided
if [ -z "$1" ]; then
    echo "No argument provided. Please provide the h5ad file name."
    exit 1
fi

# Assign the argument to a variable
h5ad_file_name=$1

# Run the python script and pipe the output to grep
python mammal/examples/scrna_cell_type/scRNA_infer.py -i "$h5ad_file_name" --test-h5ad-file --model-path ibm-research/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd --tokenizer_path ibm-research/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd  | grep -v -e "indices \[314, 315"
echo "data file $h5ad_file_name seems to be in the correct format."

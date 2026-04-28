"""
Dataset loader for cell line drug response fine-tuning using GDSC1/2 from TDC.
"""

import numpy as np
from torch.utils.data import Dataset


def sort_genes_by_value_and_name(expressions, genes):
    """
    Sort genes by expression value (descending) and name (lexical).

    Args:
        expressions: Expression values (floats) to sort by
        genes: Gene names (strings, may contain NaN) to sort

    Returns:
        Sorted list of gene names
    """
    # Convert gene names to strings to handle NaN values in TDC data
    genes_str = [str(g) if not isinstance(g, str) else g for g in genes]
    joined_vals = zip(expressions, genes_str)
    sorted_list = sorted(joined_vals, key=lambda x: (-x[0], x[1]))
    return [gene for (_, gene) in sorted_list]


class GDSCDataset(Dataset):
    """
    PyTorch Dataset for GDSC drug response data using TDC library.
    Each sample contains gene expression, drug SMILES, and IC50 values.

    Args:
        name: Dataset name - "GDSC1" or "GDSC2" (default: "GDSC2")
        path: Path to cached data (optional, TDC will download if not present)
        split: 'train', 'valid', or 'test'
        limit_samples: Limit number of samples (for testing)

    Example:
        dataset = GDSCDataset(name="GDSC2", split="train")
        sample = dataset[0] #contains: genes, expressions, drug_id, smiles, cell_line_id, Y
    """

    def __init__(self, name="GDSC2", path=None, split="train", limit_samples=None):

        from tdc.multi_pred import DrugRes

        print(f"Loading {name} dataset from TDC...")
        if path is None:
            path = "./data"
        self.tdc_data_obj = DrugRes(name=name, path=path)

        split_data = self.tdc_data_obj.get_split()
        self.data = split_data[split]

        self.gene_symbols = list(self.tdc_data_obj.get_gene_symbols())

        print(
            f"Loaded {split} split: {len(self.data)} samples with {len(self.gene_symbols)} genes"
        )

        if limit_samples:
            self.data = self.data.iloc[:limit_samples]
            print(f"Limited to {len(self.data)} samples")

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            Dictionary with keys:
                - genes: list of gene symbols
                - expressions: numpy array of expression values
                - drug_id: drug identifier
                - smiles: SMILES string
                - cell_line_id: cell line identifier
                - Y: IC50 value (target)
        """
        row = self.data.iloc[idx]

        expressions = row["Cell Line"]
        if not isinstance(expressions, np.ndarray):
            expressions = np.array(expressions)

        sample = {
            "genes": self.gene_symbols,
            "expressions": expressions,
            "drug_id": row["Drug_ID"],
            "smiles": row["Drug"],
            "cell_line_id": row["Cell Line_ID"],
            "Y": float(row["Y"]),
        }

        return sample

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    # Quick test with limited samples
    dataset = GDSCDataset(name="GDSC2", split="train", limit_samples=5)

    print(f"\nDataset length: {len(dataset)}")
    print("\nFirst sample:")
    sample = dataset[0]
    print(f"  Cell line: {sample['cell_line_id']}")
    print(f"  Drug: {sample['drug_id']}")
    print(f"  SMILES: {sample['smiles'][:50]}...")
    print(f"  Number of genes: {len(sample['genes'])}")
    print(f"  First 10 genes: {sample['genes'][:10]}")
    print(f"  IC50 (Y): {sample['Y']}")

"""
Online example to get the embeddings of MAMMAL biomedical foundation model.

This example uses the OpenAI-compatible /v1/embeddings endpoint.

Start the server first:

    vllm serve ibm-research/biomed.omics.bl.sm.ma-ted-458m \
        --runner pooling \
        --trust-remote-code \
        --tokenizer_mode mammal \
        --gpu_memory_utilization 0.4 \
        --enforce_eager \
        --no-enable-prefix-caching

Then run this script:

    python examples/online_mammal_usage.py
"""

import numpy as np
from examples.example_prompts import (
    GENE_BRCA1,
    PROTEIN_CALMODULIN,
    SMILES_ASPIRIN,
)
from openai import OpenAI


def main():
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    model_name = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"

    names = ["Calmodulin (protein)", "Aspirin (SMILES)", "BRCA1 (gene)"]
    texts = [PROTEIN_CALMODULIN, SMILES_ASPIRIN, GENE_BRCA1]

    # Pass plain text — the server tokenizes via MammalTokenizer (tokenizer_mode=mammal)
    response = client.embeddings.create(model=model_name, input=texts)

    print("=" * 100)
    print(f"{'Sequence':<30}  {'Embedding dim':>14}  {'Embedding[0]'}")
    print("=" * 100)

    embeddings = []
    for name, item in zip(names, response.data):
        emb = np.array(item.embedding)
        embeddings.append(emb)
        print(f"{name:<30}  {emb.shape[0]:>14}  {emb[0]}")


if __name__ == "__main__":
    main()

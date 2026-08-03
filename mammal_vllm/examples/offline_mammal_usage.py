"""
Offline example to get the embeddings of MAMMAL biomedical foundation model.

Usage:
    python examples/offline_mammal_usage.py
"""

import numpy as np
from examples.example_prompts import (
    GENE_BRCA1,
    PROTEIN_CALMODULIN,
    SMILES_ASPIRIN,
)
from vllm import LLM


def main():
    model_name = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"
    model = LLM(
        model=model_name,
        runner="pooling",  # use the pooling / embedding runner
        trust_remote_code=True,  # MAMMAL uses custom tokenizer code
        tokenizer_mode="mammal",  # use MAMMAL's ModularTokenizerOp via vLLM's registry
        gpu_memory_utilization=0.4,  # reduce GPU memory usage to fit in available memory
        enforce_eager=True,  # disable CUDA graphs to avoid device-side assert errors
        enable_prefix_caching=False,  # disable prefix/KV caching
    )

    names = ["Calmodulin (protein)", "Aspirin (SMILES)", "BRCA1 (gene)"]
    prompts = [PROTEIN_CALMODULIN, SMILES_ASPIRIN, GENE_BRCA1]

    outputs = model.embed(prompts)

    print("=" * 60)
    print(f"{'Sequence':<30}  {'Embedding dim':>14}")
    print("=" * 60)

    embeddings = []
    for name, output in zip(names, outputs):
        emb = np.array(output.outputs.embedding)
        embeddings.append(emb)
        print(f"{name:<30}  {emb.shape[0]:>14}")


if __name__ == "__main__":
    main()

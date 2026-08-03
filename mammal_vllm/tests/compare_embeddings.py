"""
tests/compare_embeddings.py
----------------------------
ECompare embeddings from vLLM plugin vs direct MAMMAL model.
This test requires GPU and both vllm-mammal-plugin and mammal packages installed.


Optional: set COMPARE_ONLINE=true to also compare against a running vLLM server.
To start the server::

    vllm serve ibm-research/biomed.omics.bl.sm.ma-ted-458m \
        --runner pooling \
        --trust-remote-code \
        --tokenizer_mode mammal \
        --gpu_memory_utilization 0.4 \
        --enforce_eager \
        --no-enable-prefix-caching
"""

import os
import time

import numpy as np
import torch
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from mammal.keys import (
    ENCODER_INPUTS_ATTENTION_MASK,
    ENCODER_INPUTS_TOKENS,
)
from mammal.model import Mammal
from vllm import LLM

from examples.example_prompts import (
    GENE_BRCA1,
    GENE_MALAT1,
    PROTEIN_CALMODULIN,
    PROTEIN_FLUORESCENT,
    SMILES_ASPIRIN,
    SMILES_CAFFEINE,
    SMILES_ETHER,
)

MODEL_NAME = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def get_vllm_embeddings(
    prompts: list[str],
) -> tuple[list, float, float]:
    """Get embeddings using vLLM plugin with utility functions.

    Args:
        prompts: List of text prompts to embed

    Returns:
        Tuple of (embeddings, initialization_time, inference_time)
    """
    # Time model initialization
    init_start = time.time()
    llm = LLM(
        model=MODEL_NAME,
        runner="pooling",  # use the pooling / embedding runner
        trust_remote_code=True,  # MAMMAL uses custom tokenizer code
        tokenizer_mode="mammal",  # use MammalTokenizer via vLLM's registry
        gpu_memory_utilization=0.4,  # reduce GPU memory usage to fit in available memory
        enforce_eager=True,  # disable CUDA graphs to avoid device-side assert errors
        enable_prefix_caching=False,  # disable prefix/KV caching
    )
    init_time = time.time() - init_start

    # Time inference
    inference_start = time.time()

    outputs = llm.embed(prompts)

    # Extract embeddings from outputs
    embeddings = [np.array(output.outputs.embedding) for output in outputs]

    inference_time = time.time() - inference_start

    return embeddings, init_time, inference_time


def get_online_vllm_embeddings(
    prompts: list[str], base_url: str = "http://localhost:8000/v1"
) -> tuple[list, float]:
    """Get embeddings using online vLLM server via OpenAI-compatible API.

    Args:
        prompts: List of text prompts to embed
        base_url: Base URL for the vLLM server

    Returns:
        Tuple of (embeddings, inference_time)
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package is required for online comparison. Install with: pip install openai"
        )

    client = OpenAI(base_url=base_url, api_key="EMPTY")

    # Time inference
    inference_start = time.time()

    # Pass plain text — the server tokenizes via MammalTokenizer (tokenizer_mode=mammal)
    response = client.embeddings.create(model=MODEL_NAME, input=prompts)

    # Extract embeddings from batch response
    embeddings = [np.array(data.embedding) for data in response.data]

    inference_time = time.time() - inference_start

    return embeddings, inference_time


def get_mammal_embeddings(
    model_name: str, prompts: list[str], tokenizer_op: ModularTokenizerOp | None = None
) -> tuple[list, float, float]:
    """Get embeddings using direct MAMMAL model.

    Args:
        model_name: Name of the MAMMAL model to load
        prompts: List of text prompts to embed
        tokenizer_op: Optional shared ModularTokenizerOp instance

    Returns:
        Tuple of (embeddings, initialization_time, inference_time)
    """
    # Time model initialization
    init_start = time.time()
    # Load model
    model = Mammal.from_pretrained(
        pretrained_model_name_or_path=model_name,
        allow_config_mismatch=True,
        strict=False,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)
    init_time = time.time() - init_start

    # Create tokenizer if not provided
    if tokenizer_op is None:
        tokenizer_op = ModularTokenizerOp.from_pretrained(model_name)

    # Tokenize all prompts and collect token_ids / attention_masks
    all_token_ids = []
    all_attention_masks = []
    for prompt in prompts:
        tokenized = tokenizer_op(
            {"text": prompt},
            key_in="text",
            key_out_tokens_ids="input_ids",
            key_out_attention_mask="attention_mask",
        )
        ids = tokenized["input_ids"]
        mask = tokenized["attention_mask"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        all_token_ids.append(ids)
        all_attention_masks.append(mask)

    # Pad all sequences to the same length (right-pad with 0)
    max_len = max(len(ids) for ids in all_token_ids)
    padded_ids = [ids + [0] * (max_len - len(ids)) for ids in all_token_ids]
    padded_mask = [mask + [0] * (max_len - len(mask)) for mask in all_attention_masks]

    input_ids_tensor = torch.tensor(padded_ids, dtype=torch.long).to(device)  # [B, L]
    attention_mask_tensor = torch.tensor(padded_mask, dtype=torch.long).to(
        device
    )  # [B, L]

    # Time inference only (single batched forward pass, matching vLLM)
    inference_start = time.time()

    with torch.no_grad():
        batch_dict = {
            ENCODER_INPUTS_TOKENS: input_ids_tensor,
            ENCODER_INPUTS_ATTENTION_MASK: attention_mask_tensor,
        }
        input_embeddings = model._calculate_inputs_embeddings(batch_dict)  # [B, L, D]

        encoder_output = model.t5_model.encoder(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask_tensor,
        )

        # Mean pooling over non-padding positions: [B, L, D] → [B, D]
        last_hidden_state = encoder_output.last_hidden_state  # [B, L, D]
        mask_expanded = attention_mask_tensor.unsqueeze(-1).float()  # [B, L, 1]
        pooled = (last_hidden_state * mask_expanded).sum(dim=1) / mask_expanded.sum(
            dim=1
        )  # [B, D]

        # L2-normalise each embedding
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)  # [B, D]

    embeddings = [pooled[i].cpu().numpy() for i in range(len(prompts))]

    inference_time = time.time() - inference_start

    return embeddings, init_time, inference_time


def compare_embeddings() -> None:
    """Print timing and per-prompt similarity metrics for all modalities.

    Compares:
    - vLLM offline plugin vs direct MAMMAL model (always)
    - Online vLLM server vs direct MAMMAL model (when COMPARE_ONLINE=true)
    """
    compare_online = os.environ.get("COMPARE_ONLINE", "").lower() in (
        "true",
        "1",
        "yes",
    )

    prompts = [
        PROTEIN_CALMODULIN,
        SMILES_ASPIRIN,
        SMILES_CAFFEINE,
        PROTEIN_FLUORESCENT,
        SMILES_ETHER,
        GENE_MALAT1,
        GENE_BRCA1,
    ]
    names = [
        "Calmodulin (protein)",
        "Aspirin (SMILES)",
        "Caffeine (SMILES)",
        "Fluorescent (protein)",
        "Ether (SMILES)",
        "Malat1 (gene)",
        "BRCA1 (gene)",
    ]

    # Create a single tokenizer instance shared across all MAMMAL tokenization calls
    print("\n" + "=" * 70)
    print("Creating shared tokenizer...")
    mammal_tokenizer_op = ModularTokenizerOp.from_pretrained(MODEL_NAME)

    print("\n" + "=" * 70)
    print("Getting embeddings from vLLM plugin (offline)...")
    vllm_embeddings, vllm_init_time, vllm_inference_time = get_vllm_embeddings(prompts)
    print(f"  Initialization time: {vllm_init_time:.3f}s")
    print(f"  Inference time:      {vllm_inference_time:.3f}s")
    print(f"  Total time:          {vllm_init_time + vllm_inference_time:.3f}s")

    print("\n" + "=" * 70)
    print("Getting embeddings from direct MAMMAL model...")
    mammal_embeddings, mammal_init_time, mammal_inference_time = get_mammal_embeddings(
        MODEL_NAME, prompts, mammal_tokenizer_op
    )
    print(f"  Initialization time: {mammal_init_time:.3f}s")
    print(f"  Inference time:      {mammal_inference_time:.3f}s")
    print(f"  Total time:          {mammal_init_time + mammal_inference_time:.3f}s")

    online_embeddings = None
    online_inference_time = None
    if compare_online:
        print("\n" + "=" * 70)
        print("Getting embeddings from online vLLM server...")
        try:
            online_embeddings, online_inference_time = get_online_vllm_embeddings(
                prompts
            )
            print(f"  Inference time: {online_inference_time:.3f}s")
            print("✓ Successfully retrieved online embeddings")
        except Exception as e:
            print(f"⚠ Warning: Could not get online embeddings: {e}")
            print("  Continuing with offline comparison only...")

    print("\n" + "=" * 70)
    print("Embedding Comparison Results")
    print("=" * 70)

    for i, name in enumerate(names):
        vllm_emb = vllm_embeddings[i]
        mammal_emb = mammal_embeddings[i]

        approximate_equality = np.allclose(vllm_emb, mammal_emb, atol=1e-3)
        similarity = cosine_similarity(vllm_emb, mammal_emb)
        l2_distance = np.linalg.norm(vllm_emb - mammal_emb)

        print(f"\n{name}:")
        print("  Offline vLLM comparison:")
        print(f"  vLLM shape:             {vllm_emb.shape}")
        print(f"  MAMMAL shape:           {mammal_emb.shape}")
        print(f"  Approximate equality:   {approximate_equality}")
        print(f"  Cosine similarity:      {similarity:.6f}")
        print(f"  L2 distance:            {l2_distance:.6f}")

        if online_embeddings is not None:
            online_emb = online_embeddings[i]
            online_similarity = cosine_similarity(online_emb, mammal_emb)
            online_l2 = np.linalg.norm(online_emb - mammal_emb)

            print("  Online vLLM comparison:")
            print(f"  vLLM shape:             {online_emb.shape}")
            print(f"  MAMMAL shape:           {mammal_emb.shape}")
            print(
                f"  Approximate equality:   {np.allclose(online_emb, mammal_emb, atol=1e-3)}"
            )
            print(f"  Cosine similarity:      {online_similarity:.6f}")
            print(f"  L2 distance:            {online_l2:.6f}")

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Number of prompts: {len(prompts)}\n")
    print(f"{'Method':<25} {'Init (s)':<12} {'Inference (s)':<16} {'Total (s)':<12}")
    print("-" * 65)

    vllm_total = vllm_init_time + vllm_inference_time
    print(
        f"{'vLLM (offline)':<25} {vllm_init_time:<12.3f} {vllm_inference_time:<16.3f} {vllm_total:<12.3f}"
    )

    mammal_total = mammal_init_time + mammal_inference_time
    print(
        f"{'Direct MAMMAL':<25} {mammal_init_time:<12.3f} {mammal_inference_time:<16.3f} {mammal_total:<12.3f}"
    )

    if online_inference_time is not None:
        print(
            f"{'vLLM (online)':<25} {'N/A':<12} {online_inference_time:<16.3f} {online_inference_time:<12.3f}"
        )

    print()
    print("Inference speedup (batching advantage — init is a one-time cost):")
    print("-" * 65)

    if mammal_inference_time > 0:
        vllm_speedup = mammal_inference_time / vllm_inference_time
        print(
            f"  vLLM offline vs Direct MAMMAL: {vllm_speedup:.2f}x {'faster' if vllm_speedup > 1 else 'slower'}"
            f"  (both batch all {len(prompts)} prompts in a single forward pass)"
        )

    if online_inference_time is not None and mammal_inference_time > 0:
        online_speedup = mammal_inference_time / online_inference_time
        print(
            f"  vLLM online  vs Direct MAMMAL: {online_speedup:.2f}x {'faster' if online_speedup > 1 else 'slower'}"
        )

    if online_inference_time is not None and vllm_inference_time > 0:
        online_vs_offline = vllm_inference_time / online_inference_time
        print(
            f"  vLLM online  vs vLLM offline:  {online_vs_offline:.2f}x {'faster' if online_vs_offline > 1 else 'slower'}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    compare_embeddings()

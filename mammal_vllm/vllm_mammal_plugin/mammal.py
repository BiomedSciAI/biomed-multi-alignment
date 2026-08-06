"""
vLLM-compatible encoder-only embedding model for IBM MAMMAL
(ibm-research/biomed.omics.bl.sm.ma-ted-458m).

The MAMMAL model is a T5-style encoder-decoder biomedical foundation model.
This plugin exposes only the encoder stack and returns pooled hidden states
as dense embeddings, enabling use with vLLM's pooling (embedding) runner.

Architecture notes
------------------
* MAMMAL is based on a T5 encoder-decoder transformer (458M params total).
* We use only the encoder portion and apply mean-pooling over non-padding
  token positions to produce a single embedding vector per input sequence.
* The model is registered under the architecture name ``T5ForConditionalGeneration``
  so vLLM can discover it via the ModelRegistry when the plugin is loaded.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig, T5Config, T5EncoderModel
from vllm.config import VllmConfig
from vllm.config.pooler import PoolerConfig
from vllm.model_executor.layers.pooler.seqwise import pooler_for_embed
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class MammalConfig(PretrainedConfig):
    """Configuration class for MAMMAL model."""

    model_type = "t5"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class T5ForConditionalGeneration(nn.Module):
    """
    Encoder-only wrapper around the MAMMAL T5 model for dense embeddings.

    vLLM pooling models must implement:
      * forward()  – returns last hidden states as a tensor
      * pooler()   – transforms hidden states → pooled output

    Loading is handled by vLLM's standard weight loader, which maps the
    HuggingFace ``encoder.*`` weight keys directly onto the T5EncoderModel.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        self.vllm_config = vllm_config
        model_config = vllm_config.model_config
        hf_config: T5Config = model_config.hf_config  # type: ignore[assignment]

        # Handle nested t5_config structure in MAMMAL config files
        # MAMMAL config has T5 parameters nested under 't5_config' key
        config_dict = hf_config.to_dict()
        if "t5_config" in config_dict and isinstance(config_dict["t5_config"], dict):
            # Use the nested t5_config values
            t5_config_dict = config_dict["t5_config"]
            # Create a new T5Config from the nested config
            hf_config = T5Config(**t5_config_dict)
        elif "d_model" not in config_dict:
            # If d_model is not at root level, the config is malformed
            raise ValueError(
                "MAMMAL config must have T5 parameters either at root level or under 't5_config' key. "
                f"Found keys: {list(config_dict.keys())}"
            )

        # Force encoder-only behavior in the effective config that vLLM inspects.
        # The upstream HF config advertises T5 encoder-decoder architecture, which
        # makes vLLM classify the model as encoder-decoder and route it through
        # multimodal scheduler assertions. This plugin exposes encoder-only
        # embeddings, so override those flags here.
        hf_config.use_cache = False
        hf_config.is_encoder_decoder = False
        hf_config.is_decoder = False
        hf_config.add_cross_attention = False
        hf_config.architectures = ["T5ForConditionalGeneration"]

        # -----------------------------------------------------------------
        # Build the T5 encoder.  We instantiate only the encoder stack so
        # the decoder weights are never loaded into GPU memory.
        # -----------------------------------------------------------------
        self.encoder = T5EncoderModel(hf_config)

        # Hidden dimension used for embedding output.
        self.hidden_size: int = hf_config.d_model

        # Create pooler - vLLM will use this for pooling
        self._pooler = pooler_for_embed(
            PoolerConfig(seq_pooling_type="MEAN", use_activation=False)
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        vLLM calls forward() expecting a (total_tokens, hidden_dim) tensor of
        hidden states back (no batch dim — vLLM handles batching internally
        via PoolingMetadata).

        Args:
            input_ids: Flattened 1-D tensor of shape (total_tokens,) containing
                      all token IDs from all sequences concatenated together.
            positions: Flattened 1-D tensor of shape (total_tokens,) containing
                      position indices for each token.

        Returns:
            Hidden states tensor of shape (total_tokens, hidden_dim)
        """
        # The input_ids is a 1-D concatenation of all sequences in the batch.
        # We can't process this as a single sequence because T5's self-attention
        # would allow tokens from different prompts to attend to each other.
        # Thus, we batch sequences with padding for efficient parallel processing.
        # We identify sequence boundaries by detecting position resets
        # positions looks like: [0,1,2,3,0,1,2,0,1,2,3,4,...]
        #                        ^seq1^ ^seq2^ ^seq3^
        position_resets = torch.cat(
            [
                torch.tensor(
                    [0], device=positions.device
                ),  # First token is always a start
                torch.where(positions[1:] <= positions[:-1])[0]
                + 1,  # Find where position decreases or stays same
            ]
        )

        # Split input_ids into individual sequences
        seq_starts = position_resets.tolist()
        seq_ends = seq_starts[1:] + [len(input_ids)]
        sequences = [input_ids[start:end] for start, end in zip(seq_starts, seq_ends)]

        # Find max sequence length for padding
        max_len = max(len(seq) for seq in sequences)
        batch_size = len(sequences)

        # Create padded batch tensor and attention mask
        # Use pad_token_id=0 (T5 uses 0 for padding)
        padded_input_ids = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=input_ids.device
        )
        attention_mask = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=input_ids.device
        )

        # Fill in the sequences and create attention masks
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            padded_input_ids[i, :seq_len] = seq
            attention_mask[i, :seq_len] = 1  # 1 for real tokens, 0 for padding

        # Process all sequences in a single batched forward pass
        encoder_output = self.encoder(
            input_ids=padded_input_ids,  # (batch_size, max_len)
            attention_mask=attention_mask,  # (batch_size, max_len)
        )

        # Extract hidden states and remove padding to reconstruct flattened format
        # encoder_output.last_hidden_state shape: (batch_size, max_len, hidden_dim)
        all_hidden_states = []
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            # Extract only the non-padded hidden states for this sequence
            seq_hidden_states = encoder_output.last_hidden_state[i, :seq_len, :]
            all_hidden_states.append(seq_hidden_states)

        # Concatenate all sequences back into flattened format
        hidden_states = torch.cat(
            all_hidden_states, dim=0
        )  # (total_tokens, hidden_dim)
        return hidden_states

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    @property
    def pooler(self):
        """Return the pooler instance for vLLM to use."""
        return self._pooler

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed input_ids using the encoder's embedding layer.

        This method is called by vLLM's pooling interface to get embeddings
        from input token IDs before passing them through the model.

        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_len)

        Returns:
            Embedded tokens of shape (batch_size, seq_len, hidden_size)
        """
        return self.encoder.shared(input_ids)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Map HuggingFace checkpoint keys to this module's parameter names.

        MAMMAL checkpoints store weights under the prefix ``t5_model.*``:
          - t5_model.encoder.block.0...
          - t5_model.lm_head.weight (the shared input embedding — tied to t5_model.shared)
          - t5_model.decoder.* (which we skip)

        Our model wraps T5EncoderModel as self.encoder, so parameters are:
          - encoder.encoder.block.0...
          - encoder.shared.weight

        We need to map:
          - t5_model.encoder.* → encoder.encoder.*
          - t5_model.lm_head.weight → encoder.shared.weight
        """
        weights = list(weights)  # materialise so we can scan twice

        # Find the actual embedding size from the checkpoint
        for name, tensor in weights:
            if "shared.weight" in name or "embed_tokens.weight" in name:
                actual_vocab_size = tensor.shape[0]
                current_vocab_size = self.encoder.shared.num_embeddings
                if actual_vocab_size != current_vocab_size:
                    self.encoder.resize_token_embeddings(actual_vocab_size)
                break

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Skip other task-specific heads
            if name.startswith("encoder_head.") or name.startswith("scalars_"):
                continue

            # The shared input embedding is stored as decoder.embed_tokens.weight
            # (tied weights in the full T5 model). Load it once into encoder.shared.weight.
            if name == "t5_model.decoder.embed_tokens.weight":
                param_name = "encoder.shared.weight"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(param_name)
                continue

            # Skip decoder weights and the lm_head output projection. vLLM uses MAMMAL in encoder-only mode
            # so we only load the T5 encoder weights to reduce memory usage.
            if (
                name.startswith("t5_model.decoder.")
                or name.startswith("lm_head.")
                or name == "t5_model.lm_head.weight"
            ):
                continue

            # Map checkpoint names to model parameter names:
            # t5_model.encoder.* → encoder.encoder.*
            if name.startswith("t5_model."):
                # Strip "t5_model." and prepend "encoder."
                param_name = "encoder." + name[len("t5_model.") :]

                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(param_name)

        return loaded_params

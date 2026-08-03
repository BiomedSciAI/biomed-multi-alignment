from __future__ import annotations

from pathlib import Path

from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from transformers import BatchEncoding


class MammalTokenizer:
    """vLLM-compatible tokenizer wrapper around MAMMAL's ModularTokenizerOp.

    Implements the ``TokenizerLike`` protocol so it can be registered in vLLM's
    ``TokenizerRegistry`` and used transparently whenever a text prompt is
    passed to ``LLM.embed()``.  All tokenization logic is delegated to
    ``ModularTokenizerOp``; the properties required by the protocol are derived
    from the underlying tokenizer at construction time.
    """

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *_args,
        _trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> MammalTokenizer:
        # Strip kwargs that vLLM injects but ModularTokenizerOp doesn't accept.
        modular_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "pad_token",
                "max_size",
                "on_unknown_default_value",
                "force_download",
                "resume_download",
                "proxies",
                "token",
                "local_files_only",
            )
        }
        op = ModularTokenizerOp.from_pretrained(
            str(path_or_repo_id),
            revision=revision,
            cache_dir=download_dir,
            **modular_kwargs,
        )
        return cls(op)

    def __init__(self, op: ModularTokenizerOp) -> None:
        self._op = op
        _, self._max_token_id = op.get_max_token_id()
        self._pad_token_id: int = op.get_token_id("<PAD>")
        self._eos_token_id: int = op.get_token_id("<EOS>")

    # ------------------------------------------------------------------
    # Core tokenization — called by vLLM's renderer for text prompts
    # ------------------------------------------------------------------

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> BatchEncoding:
        ids: list[list[int]] | list[int]
        if isinstance(text, list):
            ids = [self.encode(t) for t in text]
        else:
            ids = self.encode(text)
        return BatchEncoding({"input_ids": ids})

    def encode(
        self,
        text: str,
        _truncation: bool | None = None,
        _max_length: int | None = None,
        _add_special_tokens: bool = True,
    ) -> list[int]:
        sample = {"text": text}
        tokenized = self._op(sample, key_in="text", key_out_tokens_ids="input_ids")
        token_ids = tokenized["input_ids"]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return [int(x) for x in token_ids]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self._op._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------
    # TokenizerLike protocol properties
    # ------------------------------------------------------------------

    def num_special_tokens_to_add(self) -> int:
        return 0

    @property
    def all_special_tokens(self) -> list[str]:
        return list(self._op._tokenizer.get_added_vocab().keys())

    @property
    def all_special_ids(self) -> list[int]:
        added = self._op._tokenizer.get_added_vocab()
        return list(added.values())

    @property
    def bos_token_id(self) -> int:
        return self._pad_token_id  # MAMMAL has no BOS; use PAD as fallback

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return self._op.get_vocab_size()

    @property
    def max_token_id(self) -> int:
        return self._max_token_id

    @property
    def max_chars_per_token(self) -> int:
        return 10  # conservative upper bound; unused in pooling mode

    @property
    def truncation_side(self) -> str:
        return "right"

    def __hash__(self) -> int:
        return hash(id(self))

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> dict[str, int]:
        return dict(self._op._tokenizer.get_added_vocab())

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._op._tokenizer.get_added_vocab())

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._op._tokenizer.token_to_id(tokens)
        return [self._op._tokenizer.token_to_id(t) for t in tokens]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def convert_ids_to_tokens(
        self, ids, skip_special_tokens: bool = False
    ) -> list[str]:
        special_ids = set(self.all_special_ids) if skip_special_tokens else set()
        return [
            self._op._tokenizer.id_to_token(i) or ""
            for i in ids
            if i not in special_ids
        ]

    def apply_chat_template(self, messages, tools=None, **kwargs):
        raise NotImplementedError("MAMMAL does not support chat templates")

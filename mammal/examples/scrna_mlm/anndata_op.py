import io
import logging
import random
import re
from functools import partial

import numpy as np
from anndata import AnnData
from fuse.data import OpBase
from fuse.utils.ndict import NDict

from mammal.keys import SAMPLE_ID

logger = logging.getLogger(__name__)


def build_infill_query_sentinel_extension_optional(
    seq: str,
    min_sentinel_token: int,
    max_sentinel_token: int,
    decoder_start_token: str = "<DECODER_START>",
    mask_start_marker: str = "<MASK_START>",
    mask_end_marker: str = "<MASK_END>",
    sentinel_token_format: str = "<SENTINEL_ID_%>",
    behavior_when_no_masks: str = "raise",
    return_decoder_input: bool = False,
    random_order: bool = False,
    combine_consecutive_sentinel=True,
) -> tuple[dict[str, str], int]:
    """
    This function replaces all consecutive masked tokens marked with mask_start_marker and mask_end_marker, and generates matching encoder_input, decoder_input and labels

    For example:
        seq = '<banana><BANANA>ABCD<HELLO>EF<THERE>GHI<MASK_START>MNOP<MASK_END>QRST<MASK_START>U<MASK_END>VGH<MASK_START>W<MASK_END><MASK_START>Z<MASK_END>Y<MASK_START>G<MASK_END>'
        ans = build_infill_query(seq,
            min_sentinel_token = 1000,
            max_sentinel_token = 2000,
            decoder_start_token = '<DECODER_START>',
            mask_start_marker = '<MASK_START>',
            mask_end_marker = '<MASK_END>',
            sentinel_token_format = '<SENTINEL_ID_%>',
            )

        expected = dict(
            encoder_input = '<banana><BANANA>ABCD<HELLO>EF<THERE>GHI<SENTINEL_ID_1000>QRST<SENTINEL_ID_1001>VGH<SENTINEL_ID_1002>Y<SENTINEL_ID_1003>',
            decoder_input = '<DECODER_START><SENTINEL_ID_1000>MNOP<SENTINEL_ID_1001>U<SENTINEL_ID_1002>WZ<SENTINEL_ID_1003>G',
            labels = '<SENTINEL_ID_1000>MNOP<SENTINEL_ID_1001>U<SENTINEL_ID_1002>WZ<SENTINEL_ID_1003>G',
        )

    Args:
    ----
        :param min_sentinel_token:  This is the minimal allowed value for sentinel tokens
        :param max_sentinel_token:
        :param decoder_start_token:
        :param mask_start_marker:
        :param mask_end_marker:
        :param sentinel_token_format:
        :param behavior_when_no_masks: one of ['raise', 'warn', 'ignore']
            how to behave when encountering 'seq' which does not contain any <MASK_START> <MASK_END> pairs
        :param return_decoder_input:
        :param random_order: set to True to ask the model to predict the sentinels in random order (instead of from left to right)
        left to right:
        encoder_input = ABCD<SENTINEL_0>GH<SENTINEL_1>LM<SENTINEL_2>Q, labels=<SENTINEL_0>EF<SENTINEL_1>IJK<SENTINEL_2>NOP
        random:
        encoder_input = ABCD<SENTINEL_2>GH<SENTINEL_0>LM<SENTINEL_1>Q, labels=<SENTINEL_0>IJK<SENTINEL_1>NOP<SENTINEL_2>EF
        Using numpy random generator, set numpy.random.seed() if necessary
        :param combine_consecutive_sentinel:  when True, consecutive sentinels (with no tokens between them) are joined into one sentinel.
        For example:
        seq = '<banana><BANANA>ABCD<HELLO>EF<THERE>GHI<MASK_START>MNOP<MASK_END>QRST<MASK_START>U<MASK_END>VGH<MASK_START>W<MASK_END><MASK_START>Z<MASK_END>Y<MASK_START>G<MASK_END>'
        when True (the default):
        encoder_input = '<banana><BANANA>ABCD<HELLO>EF<THERE>GHI<SENTINEL_ID_1000>QRST<SENTINEL_ID_1001>VGH<SENTINEL_ID_1002>Y<SENTINEL_ID_1003>',
        when False:
        encoder_input = '<banana><BANANA>ABCD<HELLO>EF<THERE>GHI<SENTINEL_ID_1000>QRST<SENTINEL_ID_1001>VGH<SENTINEL_ID_1002><SENTINEL_ID_1003>Y<SENTINEL_ID_1004>',


        returns a tuple - first element is a dictionary that contains the keys encoder_input, decoder_input and labels, which are useful for T5 style training.
            the second element in the tuple is the next available sentinel token ID

    """
    encoder_input = io.StringIO()
    encoder_input_list: list[str] = []
    labels = io.StringIO()
    labels_list: list[str] = []

    assert max_sentinel_token > min_sentinel_token

    if ("|" in mask_start_marker) or ("|" in mask_end_marker):
        raise Exception(
            "It is not allowed to use '|' in mask_start_marker or mask_end_marker"
        )

    mask_markers_loc = list(re.finditer(f"{mask_start_marker}|{mask_end_marker}", seq))

    if behavior_when_no_masks in ["warn", "raise"]:
        if len(mask_markers_loc) == 0:
            msg = f"no mask markers provided, there won't be anything in to infill! for {seq}"
            if behavior_when_no_masks == "warn":
                logger.warn(msg)
            elif behavior_when_no_masks == "raise":
                raise Exception("ERROR: " + msg)

    # validation
    if len(mask_markers_loc) % 2 == 1:
        raise Exception("every mask_start_marker must have a matching mask_end_marker")

    prev_end_marker_end = 0
    wrote_part_outside_markers_in_last_step = True
    # go over all pairs of masks start+end
    num_sentinels = len(mask_markers_loc) // 2

    for start, end in zip(mask_markers_loc[::2], mask_markers_loc[1::2]):
        if (start.string[start.start() : start.end()] != mask_start_marker) or (
            end.string[end.start() : end.end()] != mask_end_marker
        ):
            raise Exception(
                f"Nesting {mask_start_marker} is not allowed! Each mask_start_marker must be followed by a mask_end_marker."
            )

        # parts that are not inside mask markers
        not_between_markers = seq[prev_end_marker_end : start.start()]
        wrote_part_outside_markers_in_last_step = not_between_markers != ""
        if (
            not combine_consecutive_sentinel
            or wrote_part_outside_markers_in_last_step
            or len(labels_list) == 0
        ):
            encoder_input_list.append(not_between_markers)
            # write the text between mask_start_marker and mask_end_marker
            labels_list.append(seq[start.end() : end.start()])
        else:
            labels_list[-1] = labels_list[-1] + seq[start.end() : end.start()]

        prev_end_marker_end = end.end()

    num_sentinels = len(labels_list)
    sentinel_offsets = np.arange(num_sentinels)
    if random_order:
        np.random.shuffle(sentinel_offsets)

    labels_from_sentinels: list[str] = [""] * num_sentinels
    for sentinel_offset, encoder_input_elem, labels_elem in zip(
        sentinel_offsets, encoder_input_list, labels_list
    ):
        sentinel_token = sentinel_token_format.replace(
            "%", str(min_sentinel_token + sentinel_offset)
        )
        encoder_input.write(encoder_input_elem + sentinel_token)
        labels_from_sentinels[sentinel_offset] = labels_elem

    for sentinel_offset, labels_elem in enumerate(labels_from_sentinels):
        sentinel_token = sentinel_token_format.replace(
            "%", str(min_sentinel_token + sentinel_offset)
        )
        labels.write(sentinel_token + labels_elem)

    # go over the input that is found AFTER the last mask start/end pair
    if len(mask_markers_loc) > 0:
        encoder_input.write(seq[end.end() :])

    ans = {
        "encoder_input": encoder_input.getvalue(),
        "labels": labels.getvalue(),
    }

    if return_decoder_input:
        ans["decoder_input"] = str(decoder_start_token) + ans["labels"]

    return ans, min_sentinel_token + num_sentinels


class OpMaskedSeqToMLM(OpBase):
    """

    This op use infilling.build_infill_query to replace all consecutive masked tokens marked with mask_start_marker and mask_end_marker, and generates matching encoder_input, decoder_input and labels

    For example:
        op = OpMaskedSeqToMLM(
            min_sentinel_token_id = 1000,
            max_sentinel_token_id = 2000,
            decoder_start_token = '<DECODER_START>',
            mask_start_marker = '<MASK_START>',
            mask_end_marker = '<MASK_END>',
            sentinel_token_format = '<extra_id_%>',
            )
        sample_dict = {
            "seq": '<banana><BANANA>ABCD<HELLO>EF<THERE>GHI<MASK_START>MNOP<MASK_END>QRST<MASK_START>U<MASK_END>VGH<MASK_START>W<MASK_END><MASK_START>Z<MASK_END>Y<MASK_START>G<MASK_END>'
        }

        sample_dict = op(sample_dict, key_in_masked_seq="seq", key_out="query")

        will set sample_dict to:
        sample_dict = {}
            "query.encoder_input" = '<banana><BANANA>ABCD<HELLO>EF<THERE>GHI<extra_id_1000>QRST<extra_id_1001>VGH<extra_id_1002>Y<extra_id_1003>',
            "query.decoder_input" = '<DECODER_START><extra_id_1000>MNOP<extra_id_1001>U<extra_id_1002>WZ<extra_id_1003>G',
            "query.labels" = '<extra_id_1000>MNOP<extra_id_1001>U<extra_id_1002>WZ<extra_id_1003>G',
        )

    """

    def __init__(
        self,
        min_sentinel_token: int = 1000,
        max_sentinel_token: int = 2000,
        decoder_start_token: str = "<DECODER_START>",
        mask_start_marker: str = "<MASK_START>",
        mask_end_marker: str = "<MASK_END>",
        sentinel_token_format: str = "<SENTINEL_ID_%>",
        tokenizer_type: str = "AA",
    ):
        """See bmfm_core.fm.t5.infilling.build_infill_query for details about the arguments."""
        super().__init__()
        self.build_infill_query = partial(
            build_infill_query_sentinel_extension_optional,
            min_sentinel_token=min_sentinel_token,
            max_sentinel_token=max_sentinel_token,
            decoder_start_token=decoder_start_token,
            mask_start_marker=mask_start_marker,
            mask_end_marker=mask_end_marker,
            sentinel_token_format=sentinel_token_format,
            return_decoder_input=True,
            # for GeneFormer, not combining may be better
            combine_consecutive_sentinel=True,
        )
        self.tokenizer_type = tokenizer_type

    def __call__(
        self, sample_dict: NDict, key_in_masked_seq: str, key_out: str
    ) -> None | dict | list[dict]:
        seq = sample_dict[key_in_masked_seq]
        query = self.build_infill_query(seq)
        sample_dict[key_out] = {
            "encoder_input": f"<@TOKENIZER-TYPE={self.tokenizer_type}>"
            + query[0]["encoder_input"]
            + "<EOS>",
            "decoder_input": f"<@TOKENIZER-TYPE={self.tokenizer_type}>"
            + query[0]["decoder_input"]
            + "<EOS>",
            "labels": f"<@TOKENIZER-TYPE={self.tokenizer_type}>"
            + query[0]["labels"]
            + "<EOS>",
        }
        return sample_dict


class OpRandomMaskVector(OpBase):
    """
    Operator to randomly mask a vector
    Each cell in the vector may be masked independently, with given probability.
    For genes ordered by expression, this masks one gene at a time, creating a separate label for each
    masking is IID.
    __call__ returns a tuple - first element is a dictionary that contains the keys encoder_input, decoder_input and labels, which are useful for T5 style training.
            the second element in the tuple is the next available sentinel token ID
            If the output contains no masked entries, cells are once again masked randomly, and this repeats until at least one cell is marked.

        Pretraining Format:
          Input:<@TOKENIZER-TYPE=GENE><MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENES_EXPRESSION_RANKED>[ZNF708][ZFP36]…<SENTINAL_ID_0>[ZNF3]<SENTINAL_ID_1>[ZNF4]
          Labels:<@TOKENIZER-TYPE=GENE><SENTINAL_ID_0>[YWHAB]<SENTINAL_ID_1>[CDK14]
    """

    def __init__(
        self,
        mask_prob: float,
        key_in_vec: str,
        key_out_masked_seq: str,
        mask_start_marker: str = "<MASK_START>",
        mask_end_marker: str = "<MASK_END>",
        random_seed: int = 2024,
    ):
        """
        _summary_.

        Args:
        ----
            mask_prob (float): probability to mask each cell
            key_in_vec (str): dictionary key for input
            key_out_masked_seq (str): dictionary key for output
            mask_start_marker (str, optional): string to start mask. Defaults to "<MASK_START>".
            mask_end_marker (str, optional): string to end mask. Defaults to "<MASK_END>".
            random_seed (int, optional): random seed for masking. Defaults to 2024.
        """
        super().__init__()
        self.mask_prob = mask_prob

        self.key_in_vec = key_in_vec
        self.key_out_masked_seq = key_out_masked_seq
        self.mask_start_marker = mask_start_marker
        self.mask_end_marker = mask_end_marker
        self.random = random.Random(x=random_seed)

    def _mask(self, seq):
        return self.mask_start_marker + seq + self.mask_end_marker

    def __call__(self, sample_dict: NDict) -> None | dict | list[dict]:
        seq = sample_dict[self.key_in_vec]
        assert len(seq), "can not work with empty sequences"
        masked_index: list[int] = []
        while (
            len(masked_index) == 0
        ):  # we must mask at least one value, even for short sequences
            masked_index = [
                index
                for index in range(len(seq))
                if self.random.random() < self.mask_prob
            ]
        masked_seq = [
            seq[index] if index not in masked_index else self._mask(seq[index])
            for index in range(len(seq))
        ]
        sample_dict[self.key_out_masked_seq] = (
            "<MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>"
            + ("".join(masked_seq))
        )
        return sample_dict


class OpReadAnnData(OpBase):
    """
    Op reading data from anndata.
    Each row will be added as a value to sample dict.
    """

    def __init__(
        self,
        data: AnnData | None = None,
        key_name: str = SAMPLE_ID,
        label_column: str = "label",
    ):
        """
        :param data:  input AnnData object
        :param key_name: name of value in sample_dict which will be used as the key/index
        :param label_column: name of the column which contains the label
        """
        super().__init__()

        self._key_name = key_name
        self._data = data
        self.label_column = label_column
        self.gene_names = np.array(self._data.var_names)

    def __call__(
        self, sample_dict: NDict, prefix: str | None = None
    ) -> None | dict | list[dict]:
        """
        See base class

        :param prefix: specify a prefix for the sample dict keys.
                       For example, with prefix 'data.features' and a df with the columns ['height', 'weight', 'sex'],
                       the matching keys will be: 'data.features.height', 'data.features.weight', 'data.features.sex'.
        """

        key = sample_dict[self._key_name]

        # locate the required item
        sample_dict[f"{prefix}.scrna"] = self._data[key, :].X
        sample_dict["data.label"] = self._data.obs.iloc[key].get(self.label_column)
        sample_dict[f"{prefix}.gene_names"] = self.gene_names

        return sample_dict

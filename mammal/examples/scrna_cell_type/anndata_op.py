import numpy as np
from anndata import AnnData
from fuse.data import OpBase
from fuse.utils.ndict import NDict

from mammal.keys import SAMPLE_ID


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

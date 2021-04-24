from pathlib import Path
from typing import Union

import xarray as xr

from glotaran.project.result import Result


def load_data(result: Union[xr.Dataset, Path, Result]):
    if isinstance(result, xr.Dataset):
        res = result
    else:
        if isinstance(result, Result):
            keys = list(result.data)
            res = result.data[keys[0]]
        else:
            res = xr.open_dataset(result)
    return res

from pathlib import Path
from typing import Union

import xarray as xr
from glotaran.project.result import Result


def load_data(result: Union[xr.Dataset, Path, Result]) -> xr.Dataset:
    if isinstance(result, xr.Dataset):
        return result
    elif isinstance(result, Result):
        keys = list(result.data)
        return result.data[keys[0]]
    else:
        return xr.open_dataset(result)

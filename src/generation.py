from random import randrange
from typing import List, Union

def n_d(*dims: int, low: int = 0, high: int = 10) -> Union[int, List]:
    if not dims:
        return randrange(low, high, 1)
    return [n_d(*dims[1:], low=low, high=high) for _ in range(dims[0])]

import datetime
from random import randrange
from typing import List, Union, Iterator

import numpy as np


def n_d(*dims: int, low: int = 0, high: int = 10) -> Union[int, List]:
    if not dims:
        return randrange(low, high, 1)
    return [n_d(*dims[1:], low=low, high=high) for _ in range(dims[0])]


def n_d_gen(
    *dims: int, low: int = 0, high: int = 10
) -> Union[Iterator[int], List[Iterator[int]]]:
    """
    Generate an n-dimensional array of random integers.

    Args:
        *dims(int): A list of integers representing the dimensions of the array.
        low(int): The lower bound of the random integers.
        high(int): The upper bound of the random integers.

    Returns:
        Union[Iterator[int], List[Iterator[int]]]: An iterator of random integers.
    """
    if not dims:
        yield np.random.randint(low, high=high)
    else:
        for _ in range(dims[0]):
            yield from n_d_gen(*dims[1:], low=low, high=high)


class RandomState:
    """
    Set the random state to a specific seed, then restore to the previous state on exit.
    Args:
        seed(Union[int, float, str]): The random seed.
    """

    def __init__(self, seed: Union[int, float, str]):
        self.seed = seed

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)
        return self

    def __exit__(self, *args):
        np.random.set_state(self.state)
        return False


class ArrayGeneratorND:
    """
    Create generator for n-dimensional arrays of random integers. Once initialized, the generator will always produce the same array.

    Args:
        *dims(int): A list of integers representing the dimensions of the array.
        low(int): The lower bound of the random integers.
        high(int): The upper bound of the random integers
        seed(Union[int, float, str]): Optional, sets the seed to a specific value.
    """

    def __init__(
        self,
        *dims: int,
        low: int = 0,
        high: int = 10,
        seed: Union[int, float, str] = None,
    ):
        self.dims = dims
        self.low = low
        self.high = high
        self.seed = seed or int(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) %(2**32 - 2)

    def __call__(self) -> np.ndarray:
        """
        Generate an n-dimensional array of random integers. Always generates the same array without affecting global random state.

        Returns:
            np.ndarray: An n-dimensional array of random integers
        """
        with RandomState(self.seed):
            return np.array(
                list(n_d_gen(*self.dims, low=self.low, high=self.high))
            ).reshape(self.dims)

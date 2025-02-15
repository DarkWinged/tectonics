import random as rand
from src.generation import n_d


def test_n_d():
    rand.seed(100)
    result = n_d(10)
    assert len(result) == 10
    assert min(result) >= 1

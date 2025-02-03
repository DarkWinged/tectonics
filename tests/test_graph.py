import random as rand
from src.generation import one_d


def test_one_d():
    rand.seed(100)
    result = one_d(10)
    assert len(result) == 10
    assert min(result) >= 1

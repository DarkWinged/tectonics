import random as rand
from src.generation import one_d


def test_one_d():
    expected = [1, 4, 7, 7, 7, 4, 8, 5, 0, 4]
    rand.seed(100)
    result = one_d(10)
    assert len(result) == 10
    assert result == expected

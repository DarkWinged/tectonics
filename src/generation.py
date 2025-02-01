from random import random
from typing import List


def one_d(n: int) -> List[int]:
    return [int(random() * 10) for n in range(n)]

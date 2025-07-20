import typing as tp
import numpy as np


def mark_dinucleotides(
    sequence: str, targets: tp.Union[str, tp.List[str]], mark_value: int = 1
) -> np.ndarray:
    if isinstance(targets, str):
        targets = [targets]
    output = np.zeros(len(sequence), dtype=int)
    for i in range(len(sequence) - 1):
        if sequence[i: i + 2] in targets:
            output[i] = mark_value
    return output

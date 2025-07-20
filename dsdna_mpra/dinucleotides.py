import typing as tp
import numpy as np


BASES = ['A', 'C', 'G', 'T']
BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
DINUCLEOTIDES = [
    'AT', 'TA', 'AA', 'TT', 'AC', 'GT', 'GA', 'TC',
    'CA', 'TG', 'AG', 'CT', 'CC', 'GG', 'CG', 'GC'
]
STANDARD_ORDER = [a + b for a in BASES for b in BASES]  # fast computation order
REORDER_INDEX = [STANDARD_ORDER.index(d) for d in DINUCLEOTIDES]  # biologically meaningful order


def seq_to_int_array(seq: str) -> np.ndarray:
    # Convert DNA sequence to integer array (A=0, C=1, G=2, T=3)
    return np.fromiter((BASE_TO_INT.get(base, -1) for base in seq), dtype=np.int8)


def compute_dinucleotide_counts(
    sequences: tp.Union[str, tp.Sequence[str]],
    include_reverse_complement: bool = False
) -> np.ndarray:
    if isinstance(sequences, str):
        sequences = [sequences]

    num_seqs = len(sequences)
    raw_counts = np.zeros((num_seqs, 16), dtype=np.float64)

    for i, seq in enumerate(sequences):
        nuc_arr = seq_to_int_array(seq.upper())
        nuc_arr = nuc_arr[nuc_arr >= 0]

        if len(nuc_arr) < 2:
            continue

        dinucs = 4 * nuc_arr[:-1] + nuc_arr[1:]
        uniq, counts = np.unique(dinucs, return_counts=True)
        raw_counts[i, uniq] += counts

        if include_reverse_complement:
            rc_arr = 3 - nuc_arr[::-1]
            rc_dinucs = 4 * rc_arr[:-1] + rc_arr[1:]
            uniq_rc, counts_rc = np.unique(rc_dinucs, return_counts=True)
            raw_counts[i, uniq_rc] += counts_rc

    return raw_counts[:, REORDER_INDEX]


def compute_average_dinucleotide_composition(
    sequences: tp.Union[str, tp.Sequence[str]],
    include_reverse_complement: bool = False
) -> np.ndarray:
    if isinstance(sequences, str):
        sequences = [sequences]

    raw_total_counts = np.zeros(16, dtype=np.float64)

    for seq in sequences:
        nuc_arr = seq_to_int_array(seq.upper())
        nuc_arr = nuc_arr[nuc_arr >= 0]

        if len(nuc_arr) < 2:
            continue

        dinucs = 4 * nuc_arr[:-1] + nuc_arr[1:]
        uniq, counts = np.unique(dinucs, return_counts=True)
        raw_total_counts[uniq] += counts

        if include_reverse_complement:
            rc_arr = 3 - nuc_arr[::-1]
            rc_dinucs = 4 * rc_arr[:-1] + rc_arr[1:]
            uniq_rc, counts_rc = np.unique(rc_dinucs, return_counts=True)
            raw_total_counts[uniq_rc] += counts_rc

    total = raw_total_counts.sum()
    final_counts = raw_total_counts[REORDER_INDEX]
    return final_counts / total if total > 0 else final_counts


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

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
    # convert DNA sequence to integer array (A=0, C=1, G=2, T=3)
    return np.fromiter((BASE_TO_INT.get(base, -1) for base in seq), dtype=np.int8)


def compute_dinucleotide_counts(
    sequences: tp.Union[str, tp.Sequence[str]],
    include_reverse_complement: bool = False
) -> np.ndarray:
    if isinstance(sequences, str):
        sequences = [sequences]

    int_seqs = [seq_to_int_array(seq.upper()) for seq in sequences]

    # collect valid sequences and keep their original indices
    valid_seqs_with_indices = [
        (i, arr[arr >= 0]) for i, arr in enumerate(int_seqs)
        if np.count_nonzero(arr >= 0) >= 2
    ]

    raw_counts = np.zeros((len(sequences), 16), dtype=np.float64)
    all_dinucs = []
    seq_indices = []

    for orig_idx, arr in valid_seqs_with_indices:
        dinucs = 4 * arr[:-1] + arr[1:]
        all_dinucs.append(dinucs)
        seq_indices.append(np.full(len(dinucs), orig_idx, dtype=np.int32))

        if include_reverse_complement:
            rc_arr = 3 - arr[::-1]
            rc_dinucs = 4 * rc_arr[:-1] + rc_arr[1:]
            all_dinucs.append(rc_dinucs)
            seq_indices.append(np.full(len(rc_dinucs), orig_idx, dtype=np.int32))

    if all_dinucs:
        all_dinucs_flat = np.concatenate(all_dinucs)
        all_indices_flat = np.concatenate(seq_indices)
        flat_ids = all_indices_flat * 16 + all_dinucs_flat
        uniq, counts = np.unique(flat_ids, return_counts=True)

        rows = uniq // 16
        cols = uniq % 16
        raw_counts[rows, cols] += counts

    return raw_counts[:, REORDER_INDEX]


def compute_average_dinucleotide_composition(
    sequences: tp.Union[str, tp.Sequence[str]],
    include_reverse_complement: bool = False
) -> np.ndarray:
    if isinstance(sequences, str):
        sequences = [sequences]

    seq_arrays = [seq_to_int_array(seq.upper()) for seq in sequences]
    seq_arrays = [arr[arr >= 0] for arr in seq_arrays]  # consider only A, C, G, T
    seq_arrays = [arr for arr in seq_arrays if len(arr) >= 2]  # filter short sequences

    if not seq_arrays:
        return np.zeros(16, dtype=np.float64)

    forward_dinucs = [4 * arr[:-1] + arr[1:] for arr in seq_arrays]

    if include_reverse_complement:
        reverse_dinucs = [
            4 * (3 - arr[::-1][:-1]) + (3 - arr[::-1][1:])
            for arr in seq_arrays
        ]
        all_dinucs = np.concatenate(forward_dinucs + reverse_dinucs)
    else:
        all_dinucs = np.concatenate(forward_dinucs)

    uniq, counts = np.unique(all_dinucs, return_counts=True)
    total_counts = np.zeros(16, dtype=np.float64)
    total_counts[uniq] = counts

    final_counts = total_counts[REORDER_INDEX]
    total = final_counts.sum()

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

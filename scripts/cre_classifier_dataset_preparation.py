from pathlib import Path
import typing as tp
import os
import sys
import subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm
from ushuffle import shuffle

sys.path.insert(0, '..')
from dsdna_mpra import config, boda2  # noqa E402


ENCODE_FILES = {
    'ENCFF379UDA': 'promoter-like',
    'ENCFF036NSJ': 'proximal',
    'ENCFF535MKS': 'distal',
    'ENCFF262LCI': 'ctcf-only'
}

VALID_BASES = {'A', 'C', 'G', 'T'}

BATCH_SIZE = 500


def run_command(cmd: str) -> tp.Tuple[str, str]:
    """Run a shell command and return stdout and stderr."""
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True
    )
    stdout, stderr = process.communicate()
    return stdout, stderr


def extract_info_field(info_str: str, key: str) -> str:
    """Extract a specific field from a GTF info string."""
    for entry in info_str.split(';'):
        if entry.strip().startswith(key):
            return entry.split()[1].strip('"')
    return ''


def collect_genome_sequences(
    genome_record: tp.Dict[str, str],
    chromosomes: tp.Iterable[str],
    positions: tp.Iterable[int],
    region_size: int
) -> tp.List[str]:
    """Extract sequences of given region size centered around positions."""
    sequences = []
    for chrom, pos in zip(chromosomes, positions):
        chrom_seq = genome_record[chrom]
        half_size = region_size // 2
        start = max(0, pos - half_size)
        end = min(len(chrom_seq), pos + half_size)
        sequences.append(chrom_seq[start:end])
    return sequences


def annotate_encode_peak_file(
    genome_record: tp.Dict[str, str],
    encode_id: str,
    region_type: str,
    homer_script: Path,
    genome_path: Path,
    gtf_path: Path,
    strand_lookup_df: pd.DataFrame,
    input_dir: Path,
    output_dir: Path
) -> None:
    """Annotate ENCODE peak BED file using HOMER and GENCODE."""
    input_file = input_dir / f"{encode_id}.bed"
    output_file = output_dir / f"{encode_id}_gencode_v46.csv"

    cmd = f"{homer_script} {input_file} {genome_path} -gtf {gtf_path} > {output_file}"
    run_command(cmd)

    annot_df = pd.read_csv(output_file, sep='\t')
    annot_df['gencode_annotation'] = annot_df['Annotation'].str.split('(').str[0].str.strip()
    annot_df['encode_region_type'] = region_type
    annot_df['original_index'] = annot_df.iloc[:, 0]

    annot_df.rename(columns={
        'Chr': 'chromosome',
        'Start': 'begin',
        'End': 'end',
        'Distance to TSS': 'distance_to_tss',
        'Nearest PromoterID': 'nearest_promoter_id',
        'Gene Name': 'gene_name',
        'Gene Type': 'gene_type'
    }, inplace=True)

    annot_df = annot_df[[
        'original_index', 'chromosome', 'begin', 'end', 'encode_region_type',
        'gencode_annotation', 'distance_to_tss', 'gene_name', 'gene_type', 'nearest_promoter_id'
    ]].rename(columns={'original_index': 'encode_id'})

    bed_df = pd.read_csv(input_file, sep='\t', header=None, usecols=[0, 1, 2, 3, 9])
    bed_df.columns = ['chromosome', 'begin', 'end', 'encode_id', 'encode_properties']

    index_map = dict(zip(bed_df.encode_id.values, bed_df.index))
    annot_df = annot_df.sort_values('encode_id', key=np.vectorize(index_map.get)).reset_index(drop=True)

    transcript_ids = annot_df['nearest_promoter_id'].unique()
    relevant_strands = strand_lookup_df[strand_lookup_df['transcript_id'].isin(transcript_ids)]
    strand_map = dict(relevant_strands.values)
    annot_df['nearest_promoter_strand'] = annot_df['nearest_promoter_id'].map(strand_map)

    # ensure input BED file and annotation agree
    assert all(annot_df['chromosome'] == bed_df['chromosome']), "Chromosome mismatch"
    assert all(annot_df['begin'] == bed_df['begin'] + 1), "Start position mismatch"

    # add central sequence of 300 bp
    center_positions = annot_df[['begin', 'end']].mean(axis=1).astype(int)
    annot_df['sequence'] = collect_genome_sequences(
        genome_record,
        annot_df['chromosome'],
        center_positions,
        300
    )

    annot_df.to_csv(output_file, index=False)


def load_genome_fasta(fasta_path: Path) -> tp.Dict[str, str]:
    """Load genome sequences from FASTA file into a dictionary."""
    genome = {}
    header = None
    with fasta_path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                header = line[1:]
                genome[header] = []
            else:
                genome[header].append(line)
    return {k: ''.join(v) for k, v in genome.items()}


def annotate_all_encode_peaks() -> None:
    gtf_path = config.RAW_DIR / 'gencode.v46.annotation.gtf'
    homer_script = config.PATH_TO_HOMER / 'bin/annotatePeaks.pl'
    genome_path = config.PATH_TO_HOMER / 'data/genomes/hg38'
    input_dir = config.RAW_DIR / 'encode'
    output_dir = config.PROCESSED_DIR / 'encode'
    fasta_path = config.RAW_DIR / "hg38.fasta"

    output_dir.mkdir(exist_ok=True)
    os.environ["PATH"] += os.pathsep + str((config.PATH_TO_HOMER / 'bin').resolve())

    gtf_columns = [
        'chromosome', 'source', 'feature', 'start', 'stop',
        '.', 'strand', 'level', 'attributes'
    ]
    gtf_df = pd.read_csv(gtf_path, sep='\t', skiprows=5, header=None, names=gtf_columns)
    gtf_df['transcript_id'] = gtf_df['attributes'].apply(lambda attr: extract_info_field(attr, 'transcript_id'))
    strand_df = gtf_df[['transcript_id', 'strand']]

    genome = load_genome_fasta(fasta_path)
    for encode_id, region_type in tqdm(ENCODE_FILES.items(), desc="Annotating ENCODE files"):
        annotate_encode_peak_file(
            genome_record=genome,
            encode_id=encode_id,
            region_type=region_type,
            homer_script=homer_script,
            genome_path=genome_path,
            gtf_path=gtf_path,
            strand_lookup_df=strand_df,
            input_dir=input_dir,
            output_dir=output_dir
        )


def run_malinois_predictions() -> None:
    """Run Malinois model predictions on annotated ENCODE regions."""
    boda2.unpack_model_artifact(
        config.RAW_DIR / 'malinois_artifacts__20211113_021200__287348.tar.gz',
        config.MALINOIS_MODEL_DIR
    )
    malinois_model = boda2.load_malinois_model(config.MALINOIS_MODEL_DIR)

    for encode_id in ENCODE_FILES:
        csv_path = config.PROCESSED_DIR / f"encode/{encode_id}_gencode_v46.csv"
        encode_df = pd.read_csv(csv_path)
        # extract central 200 bp from 300 bp central sequence
        sequences = encode_df['sequence'].str.slice(50, 250)
        predictions = boda2.compute_malinois_model_predictions(
            malinois_model,
            tile_sequences=sequences,
            batch_size=BATCH_SIZE,
            use_tqdm=True
        )
        predictions_df = pd.DataFrame(
            predictions,
            columns=['malinois_k562_lfc', 'malinois_hepg2_lfc', 'malinois_sknsh_lfc']
        )
        encode_df = pd.concat([encode_df, predictions_df], axis=1)
        encode_df.to_csv(csv_path, index=False)


def add_shuffled_sequences() -> None:
    K_LET = 3
    for encode_id in tqdm(ENCODE_FILES, desc="Annotating ENCODE files"):
        csv_path = config.PROCESSED_DIR / f"encode/{encode_id}_gencode_v46.csv"
        encode_df = pd.read_csv(csv_path)
        sequences = encode_df['sequence'].str.slice(50, 250)
        encode_df['sequence_shuffled'] = [
            shuffle(seq.encode("utf-8"), K_LET).decode("utf-8")
            for seq in sequences
        ]
        encode_df.to_csv(csv_path, index=False)


def merge_encode_datasets() -> None:
    # Merge and filter ENCODE datasets based on Malinois activity thresholds across selected cell types.
    malinois_thresholds = pd.read_csv(config.RESULTS_DIR / 'thresholds_malinois_log2_1p.csv')
    malinois_thresholds = dict(malinois_thresholds[['cell', 'threshold']].values)
    merged = []
    for encode_index, encode_id in enumerate(ENCODE_FILES):
        encode_df = pd.read_csv(config.PROCESSED_DIR / f"encode/{encode_id}_gencode_v46.csv")
        is_active = np.logical_or.reduce([
            encode_df[f'malinois_{cell}_lfc'] >= malinois_thresholds[cell]
            for cell in ['k562', 'hepg2', 'sknsh']
        ])
        encode_df = encode_df[is_active]
        encode_df['real_class_index'] = 2 * encode_index
        encode_df['shuff_class_index'] = 2 * encode_index + 1
        merged.append(encode_df)
    pd.concat(merged, ignore_index=True).to_csv(config.PROCESSED_DIR / "encode/cre_classifier_dataset.csv", index=False)


def augment_with_reverse_complement(seqs_oh: np.ndarray, flanked: bool) -> np.ndarray:
    # Returns original and reverse-complemented one-hot sequences concatenated
    revcomp = seqs_oh.copy()
    if flanked:
        revcomp[..., 200:400] = revcomp[..., 200:400][..., ::-1, ::-1]
    else:
        revcomp = revcomp[..., ::-1, ::-1]
    return np.concatenate([seqs_oh, revcomp], axis=0)


def encode_dataframe_to_dataset(encode_df: pd.DataFrame, output_path: Path) -> None:

    # pick encode records with valid sequences
    valid_mask = np.array([all(base in VALID_BASES for base in seq) for seq in encode_df.sequence.values])
    encode_df = encode_df[valid_mask]

    # onehot-encoding and MPRA flanks attachment
    real_onehots = np.array([boda2.dna2tensor(seq).cpu().numpy() for seq in encode_df.sequence.values])
    _, real_onehots_flanked = boda2.convert_sequences_to_malinois_input(encode_df.sequence.str.slice(50, 250))
    real_onehots_flanked = real_onehots_flanked.cpu().numpy()
    _, shuff_onehots_flanked = boda2.convert_sequences_to_malinois_input(encode_df.sequence_shuffled)
    shuff_onehots_flanked = shuff_onehots_flanked.cpu().numpy()
    real_class = encode_df.real_class_index.values
    shuff_class = encode_df.shuff_class_index.values

    # add reverse complements
    real_onehots = augment_with_reverse_complement(real_onehots, flanked=False)
    real_onehots_flanked = augment_with_reverse_complement(real_onehots_flanked, flanked=True)
    shuff_onehots_flanked = augment_with_reverse_complement(shuff_onehots_flanked, flanked=True)

    # dataset
    dataset = dict()
    dataset['sequence_with_genome_context'] = np.repeat(real_onehots, 2, axis=0)
    dataset['sequence_with_mpra_flanks'] = np.concatenate([real_onehots_flanked, shuff_onehots_flanked])
    dataset['class'] = np.concatenate([real_class, real_class, shuff_class, shuff_class])
    tss_distance_log10p = np.log10(np.abs(encode_df.distance_to_tss.values) + 1)
    dataset['tss_distance_log10p'] = np.tile(tss_distance_log10p, 4)
    tss_strand = [1 if strand == '+' else -1 for strand in encode_df.nearest_promoter_strand.values]
    dataset['tss_strand'] = np.tile(tss_strand, 4)
    dataset['sequence_is_shuffled'] = np.repeat([False, True], 2 * encode_df.shape[0])
    dataset['original_index'] = np.tile(encode_df.index.values, 4)

    # save
    np.savez(output_path, **dataset)


def create_train_test_datasets(
    train_fraction: float = 0.9,
    split_seed: int = 42
) -> None:
    assert 0 < train_fraction < 1, "train_fraction must be between 0 and 1"
    rng = np.random.default_rng(split_seed)
    encode_df = pd.read_csv(config.PROCESSED_DIR / "encode/cre_classifier_dataset.csv")
    shuffle_idxs = np.arange(encode_df.shape[0])
    rng.shuffle(shuffle_idxs)
    split = int(train_fraction * encode_df.shape[0])
    # train dataset
    encode_train = encode_df.iloc[shuffle_idxs[:split]]
    train_path = config.PROCESSED_DIR / "encode/cre_classifier_train_dataset.npz"
    encode_dataframe_to_dataset(encode_train, train_path)
    print(f"Train dataset saved to: {train_path}")
    # test dataset
    encode_test = encode_df.iloc[shuffle_idxs[split:]]
    test_path = config.PROCESSED_DIR / "encode/cre_classifier_test_dataset.npz"
    encode_dataframe_to_dataset(encode_test, test_path)
    print(f"Test dataset saved to: {test_path}")


if __name__ == "__main__":
    # annotate_all_encode_peaks()
    # run_malinois_predictions()
    # add_shuffled_sequences()
    merge_encode_datasets()
    create_train_test_datasets()

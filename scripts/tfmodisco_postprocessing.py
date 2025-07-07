import sys
from pathlib import Path
import typing as tp

import h5py
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from numpy.typing import NDArray

sys.path.insert(0, '..')
from dsdna_mpra import config, motifs  # noqa E402


def postprocess_modisco_report() -> None:
    summary_path = config.RAW_DIR / 'scenic_human_motif_collection_summary.csv'
    report_path = config.PROCESSED_DIR / "malinois_K562_modisco_report/motifs.html"

    # load PWM ID to gene mapping
    scenic_summary = pd.read_csv(summary_path)
    pwm_to_gene = dict(scenic_summary[['pwm_id', 'genes']].values)
    pwm_ids_to_genes = np.vectorize(pwm_to_gene.get)

    # load HTML table as DataFrame, add gene name columns
    report_df = pd.read_html(report_path)[0]
    for idx in range(3):
        report_df[f"gene{idx}"] = pwm_ids_to_genes(report_df[f"match{idx}"])

    # restore image sources lost during `pd.read_html`
    with open(report_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, "html.parser")

    img_columns = ['modisco_cwm_fwd', 'modisco_cwm_rev', 'match0_logo', 'match1_logo', 'match2_logo']
    img_tags = soup.find_all('img')
    img_sources = [img['src'] for img in img_tags]
    report_df[img_columns] = np.array(img_sources).reshape(report_df.shape[0], len(img_columns))

    # define image rendering for HTML display
    def path_to_image_html(path):
        return f'<img src="{path}" width="240">'

    report_df = report_df[[
        'pattern', *img_columns, 'gene0', 'gene1', 'gene2',
        'num_seqlets', 'match0', 'match1', 'match2', 'qval0', 'qval1', 'qval2'
    ]]
    output_path = config.PROCESSED_DIR / "malinois_K562_modisco_report/motifs_with_logos.html"
    report_df.to_html(output_path, escape=False, formatters={col: path_to_image_html for col in img_columns})


def iterate_modisco_h5_output(h5_file: Path) -> tp.Generator[tuple[str, str, h5py.Group], None, None]:
    with h5py.File(h5_file, "r") as f:
        allowed_keys = {"pos_patterns", "neg_patterns"}
        unexpected_keys = set(f.keys()) - allowed_keys
        if unexpected_keys:
            raise ValueError(f"Unexpected keys in HDF5 file: {unexpected_keys}")
        for pattern_type in f:
            data_group = f[pattern_type]
            for pattern_id in data_group:
                yield pattern_type, pattern_id, data_group[pattern_id]


def trim_tfmodisco_pattern(pattern: h5py.Group, trim_thresh: float) -> tuple[int, int]:
    """
    Trim a pattern based on a threshold applied to the absolute sum of hypothetical contributions.

    Returns the start and end indices (inclusive-exclusive) of the trimmed region.
    """
    if not (0 <= trim_thresh <= 1):
        raise ValueError("`trim_thresh` must be a float between 0 and 1.")

    contribs: NDArray = pattern["hypothetical_contribs"][:]
    position_scores = np.sum(np.abs(contribs), axis=1)
    threshold = np.max(position_scores) * trim_thresh
    passing_indices = np.where(position_scores >= threshold)[0]

    if passing_indices.size == 0:
        return 0, contribs.shape[0]  # no trimming possible

    start, end = np.min(passing_indices), np.max(passing_indices) + 1
    return start, end


def load_tfmodisco_motifs(h5_file: Path, motif_type: str, trim_thresh: float) -> dict[str, NDArray]:
    """
    Extract and optionally trim motifs from a MoDISco HDF5 file.

    Parameters:
    - h5_file: Path to the HDF5 file.
    - motif_type: One of 'pwm', 'pfm', or other (interpreted as contrib-based).
    - trim_thresh: Fraction threshold for trimming.

    Returns:
    - Dictionary mapping pattern IDs to trimmed motif arrays (shape: 4 x L).
    """
    motifs_dct = {}

    for pattern_sign, pattern_id, pattern_group in iterate_modisco_h5_output(h5_file):
        full_pattern_id = f"{pattern_sign[:4]}{pattern_id}"

        if motif_type in {"pwm", "pfm"}:
            motif = pattern_group["sequence"][:]
        else:
            motif = pattern_group["hypothetical_contribs"][:]

        start, end = trim_tfmodisco_pattern(pattern_group, trim_thresh)
        trimmed_motif = motif[start:end].T  # shape: 4 x L

        if motif_type == "pwm":
            pwm = motifs.pfm_to_pwm(trimmed_motif)
            pwm -= pwm.max(axis=0)
            motifs_dct[full_pattern_id] = pwm
        else:
            motifs_dct[full_pattern_id] = trimmed_motif

    return motifs_dct


def collect_k562_mpra_tf_motifs() -> None:

    combined_cwms = {}

    # load CWMs identified using TF-MoDISco-lite
    modisco_path = config.PROCESSED_DIR / "malinois_K562_modisco_results.h5"
    modisco_cwms = load_tfmodisco_motifs(modisco_path, motif_type='cwm', trim_thresh=0.1)
    combined_cwms.update(modisco_cwms)

    # load E2F PWMs from SCENIC collection and convert them to CWMs
    scenic_summary_path = config.RAW_DIR / 'scenic_human_motif_collection_summary.csv'
    scenic_pwm_dir = config.RAW_DIR / 'scenic_v10nr_clust_public_singletons'
    scenic_summary_df = pd.read_csv(scenic_summary_path)

    e2f_pwm_ids = [
        'taipale_cyt_meth__E2F1_NWTTTGGCGCCAWWWN_FL',
        'taipale_cyt_meth__E2F2_GCGCGCGCGYW_eDBD_repr'
    ]

    filtered_summary = scenic_summary_df[scenic_summary_df.pwm_id.isin(e2f_pwm_ids)]
    for file_name, pwm_id, pwm_index in filtered_summary[['file', 'pwm_id', 'pwm_index']].values:
        pwm_file_path = scenic_pwm_dir / file_name
        pwm = motifs.parse_pwm_file(pwm_file_path)[pwm_id]
        combined_cwms[f'scenic_pwm_{pwm_index}'] = motifs.pfm_to_cwm(pwm)

    # load 3' splice site PWM from Mersch et al. (BMC Bioinformatics, 2008)
    splice_site_path = config.RAW_DIR / 'splice_site_pwms.cb'
    splice_pwm = motifs.parse_pwm_file(splice_site_path)['3ss']
    combined_cwms['mersch_2008_3ss'] = motifs.pfm_to_cwm(splice_pwm)

    # include forward and reverse-complement CWMs
    final_cwms = {}
    for motif_name, cwm in combined_cwms.items():
        final_cwms[f"{motif_name}_fwd"] = cwm
        final_cwms[f"{motif_name}_rev"] = cwm[::-1, ::-1]

    # write final CWM dictionary to file
    output_path = config.RESULTS_DIR / 'malinois_K562_tf_motifs.cb'
    motifs.write_pwm_file(final_cwms, output_path)


if __name__ == "__main__":
    postprocess_modisco_report()
    collect_k562_mpra_tf_motifs()

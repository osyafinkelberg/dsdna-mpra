import typing as tp
import sys
import shutil

import numpy as np
import pandas as pd
from Bio import Entrez, SeqIO

import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm

sys.path.insert(0, '..')
from dsdna_mpra import config, dinucleotides, plots  # noqa E402


def collect_plot_data(
    genomes_summary: pd.DataFrame,
    paired_tiles: pd.DataFrame,
    virus_predictions: pd.DataFrame,
    genbank_cds: pd.DataFrame,
    genome: str
) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collects and computes various genome-wide matrices for plotting:
    - Relative tile activity ranks across cell lines
    - GenBank CDS annotation matrix
    - Dinucleotide composition matrix
    - Predicted CRE type matrix

    Returns:
        Tuple of (activity_matrix, cds_matrix, dinucleotide_matrix, cre_type_matrix)
    """
    genome_size = genomes_summary.loc[
        genomes_summary.accession_id == genome, 'genome_size'
    ].squeeze()

    # === activity matrix (tile activity rank per cell line) ===
    activity_matrix = np.full((len(config.CELL_LINES), genome_size), -1, dtype=np.float64)
    virus_tiles = paired_tiles[paired_tiles.genome == genome]

    for cell_idx, cell in enumerate(config.CELL_LINES):
        cell_tiles = virus_tiles[virus_tiles.cell == cell]
        for begin, rel_rank in cell_tiles[['begin', 'cell_rank_relative']].itertuples(index=False):
            end = min(begin + 200, genome_size)
            current_vals = activity_matrix[cell_idx, begin: end]
            activity_matrix[cell_idx, begin: end] = np.maximum(current_vals, rel_rank)

    # === CDS matrix (strand-specific CDS regions) ===
    cds_matrix = np.zeros((2, genome_size), dtype=int)
    vir_cds = genbank_cds[genbank_cds.genome == genome]

    for begin, end, strand in vir_cds[['begin', 'end', 'strand']].itertuples(index=False):
        row = 0 if strand == '+' else 1
        cds_matrix[row, begin:end] += 1

    # mask regions with no CDS annotations
    cds_matrix[:, cds_matrix.sum(axis=0) == 0] = -1

    # === dinucleotide matrix ===
    Entrez.email = config.ENTREZ_EMAIL
    with Entrez.efetch(db="nucleotide", id=genome, rettype="fasta", retmode="text") as handle:
        genome_seq = SeqIO.read(handle, "fasta").seq

    cg_cts = dinucleotides.mark_dinucleotides(genome_seq, ['CG', 'CC', 'CG', 'GC'], mark_value=1)
    at_cts = dinucleotides.mark_dinucleotides(genome_seq, ['AA', 'AT', 'TA', 'TT'], mark_value=2)
    dinucleotide_matrix = np.stack([cg_cts, at_cts])

    # === CRE type matrix ===
    cre_type_matrix = np.full(genome_size, fill_value=len(config.ENCODE_CRE_TYPES_SHORT), dtype=int)
    cre_preds = virus_predictions[virus_predictions.genome == genome]

    for begin, cre_class in cre_preds[['begin', 'predicted_class_short']].itertuples(index=False):
        class_idx = config.ENCODE_CRE_TYPES_SHORT.index(cre_class)
        end = min(begin + 200, genome_size)
        cre_type_matrix[begin:end] = class_idx

    return activity_matrix, cds_matrix, dinucleotide_matrix, cre_type_matrix


def plot_mpra_activity_genomewide(
    genomes_summary: pd.DataFrame,
    paired_tiles: pd.DataFrame,
    virus_predictions: pd.DataFrame,
    genbank_cds: pd.DataFrame,
    genome: str
) -> plt.Figure:
    """
    Plot genome-wide MPRA activity and genomic annotations for a given viral genome.

    Returns:
        A matplotlib Figure object with 5 stacked tracks:
            - MPRA activity across cell lines
            - Number of active cells per tile
            - CDS regions by strand
            - Dinucleotide enrichment
            - Predicted CRE types
    """
    activity_matrix, cds_matrix, dinucleotide_matrix, cre_type_matrix = collect_plot_data(
        genomes_summary, paired_tiles, virus_predictions, genbank_cds, genome
    )
    fig, axes = plt.subplots(
        figsize=(30, 12),
        nrows=5,
        height_ratios=[2, 0.5, 0.5, 0.5, 0.5]
    )

    # === Track 1: MPRA activity ===
    ax = axes[0]
    virus_label = ', '.join(
        genomes_summary.loc[genomes_summary.accession_id == genome, ['family', 'strain']].iloc[0]
    )
    ax.set_title(virus_label, fontsize=30)
    activity_img = ax.imshow(activity_matrix, aspect="auto", cmap='RdBu_r', vmin=-.4, vmax=.4, interpolation="none")
    ax.set_ylabel('MPRA\nCell rank', fontsize=20)
    ax.set_yticks(np.arange(len(config.CELL_LINES)))
    ax.set_yticklabels(config.CELL_LINES)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.grid(False)
    cbar_ax = fig.add_axes([.91, 0.72, 0.03, 0.15])
    fig.colorbar(activity_img, cax=cbar_ax, label='Centered rank')

    # === Track 2: Number of active cells ===
    ax = axes[1]
    n_active = (activity_matrix >= 0).sum(axis=0)
    active_img = ax.imshow(n_active[None, :], aspect="auto", cmap='Reds', vmin=0, vmax=6, interpolation="none")
    ax.set_ylabel('# Active\nCells', fontsize=20, rotation=0)
    ax.yaxis.set_label_coords(-0.06, 0.25)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.grid(False)
    cbar_ax = fig.add_axes([.91, 0.67, 0.03, 0.04])
    cbar = fig.colorbar(active_img, cax=cbar_ax, ticks=[0, 3, 6])
    cbar.set_ticklabels([0, 3, 6])

    # === Track 3: CDS regions ===
    ax = axes[2]
    cds_cmap = colors.ListedColormap(['white', 'lightgrey', 'darkgrey', 'black'])
    cds_img = ax.imshow(cds_matrix, aspect="auto", cmap=cds_cmap, vmin=-1, vmax=2, interpolation="none")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['+ strand', '- strand'])
    ax.set_ylabel('CDS', fontsize=20)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.grid(False)
    cbar_ax = fig.add_axes([.91, 0.62, 0.03, 0.04])
    cbar = fig.colorbar(cds_img, cax=cbar_ax, ticks=np.arange(4) - 1)
    cbar.set_ticklabels(['No', '1', '1 (strand)', '>1'])

    # === Track 4: Dinucleotide frequencies ===
    ax = axes[3]
    dinuc_cmap = colors.ListedColormap(['white', 'firebrick', 'cornflowerblue'])
    _ = ax.imshow(dinucleotide_matrix, aspect="auto", cmap=dinuc_cmap, interpolation="none")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['CG/GC/CC', 'AA/AT/TA/TT'])
    ax.set_ylabel('Dinucleotides', fontsize=20)
    ax.yaxis.set_label_coords(-0.075, 0.75)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.grid(False)

    # === Track 5: Predicted CRE types ===
    ax = axes[4]
    cre_labels = config.ENCODE_CRE_TYPES_SHORT + ['Not active']
    cre_cmap = colors.ListedColormap(['firebrick', 'orange', 'cornflowerblue', 'forestgreen', 'darkgrey', 'white'])
    cre_img = ax.imshow(
        cre_type_matrix[None, :], aspect="auto", cmap=cre_cmap,
        vmin=0, vmax=len(cre_labels) - 1, interpolation="none"
    )
    ax.set_ylabel('CRE type', rotation=0, fontsize=20)
    ax.yaxis.set_label_coords(-0.07, 0.25)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.grid(False)
    cbar_ax = fig.add_axes([.91, 0.56, 0.03, 0.05])
    cbar = fig.colorbar(cre_img, cax=cbar_ax, ticks=np.arange(len(cre_labels)))
    cbar.set_ticklabels(cre_labels)

    plt.subplots_adjust(hspace=0.01)
    return fig


def main() -> None:
    # === load data ===
    virus_genomes = pd.read_csv(config.RAW_DIR / 'virus_genbank_ids.txt').columns.values
    genomes_summary = pd.read_csv(config.PROCESSED_DIR / 'summary_virus_genome_records.csv')
    genomes_summary = genomes_summary[genomes_summary.accession_id.isin(virus_genomes)]

    paired_tiles = pd.read_csv(config.RESULTS_DIR / "virus_paired_tiles_cds_overlap.csv")
    threshold_ranks = [
        {
            'cell': cell,
            'threshold_rank': group.iloc[np.argmin(np.abs(group.tile_lfc - group.threshold))].cell_rank
        }
        for cell, group in paired_tiles.groupby('cell', observed=True)
    ]
    paired_tiles = paired_tiles.merge(pd.DataFrame(threshold_ranks), on='cell', how='left')
    paired_tiles['cell_rank_relative'] = paired_tiles.cell_rank - paired_tiles.threshold_rank

    virus_predictions = pd.read_csv(config.RESULTS_DIR / 'k562_active_tiles_classification.csv')
    virus_predictions['predicted_class_short'] = np.where(
        virus_predictions.predicted_class.str.contains('shuffled'),
        'Undetermined', virus_predictions.predicted_class,
    )
    virus_predictions = paired_tiles[paired_tiles.cell == 'K562'].merge(
        virus_predictions[['tile_id', 'predicted_class_short']], on='tile_id', how='right'
    )
    genbank_cds = pd.read_csv(config.PROCESSED_DIR / 'virus_cds_positions.csv')

    # === generate genome-wide maps for all viruses ===
    temp_dir = config.FIGURES_DIR / 'temp_gw_maps'
    temp_dir.mkdir(exist_ok=True)

    for figure_idx, genome in tqdm(
        enumerate(genomes_summary.accession_id),
        total=genomes_summary.shape[0],
        desc='plot genome-wide maps',
        file=sys.stdout
    ):
        fig = plot_mpra_activity_genomewide(
            genomes_summary=genomes_summary, paired_tiles=paired_tiles,
            virus_predictions=virus_predictions, genbank_cds=genbank_cds,
            genome=genome
        )
        fig.savefig(temp_dir / f"{'{0:0=2d}'.format(figure_idx)}_{genome}.jpg", dpi=300, format='jpg')
        plt.close()

    plots.convert_jpgs_to_pdf(temp_dir, config.FIGURES_DIR / 'fig_7A_genomewide_maps.pdf')
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()

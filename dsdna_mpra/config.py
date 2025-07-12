from pathlib import Path
import os


# base project directory (set manually)
BASE_DIR = Path("/projectnb/vtrs/joseff/dsdna-mpra")
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "bioinfo.project@gmail.com")

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = DATA_DIR / "figures"
MALINOIS_MODEL_DIR = RAW_DIR / "malinois_model/artifacts"

CELL_LINES = ['GM12878', 'Jurkat', 'MRC5', 'A549', 'HEK293', 'K562']
CELL_LINE_COLORS = ['navy', 'cornflowerblue', 'forestgreen', 'deeppink', 'orangered', 'firebrick']

DSDNA_FAMILIES = ['Polyomaviridae', 'Papillomaviridae', 'Adenoviridae', 'Herpesviridae']
DSDNA_FAMILY_COLORS = {
    'Polyomaviridae': 'deepskyblue',
    'Papillomaviridae': 'blueviolet',
    'Adenoviridae': 'orange',
    'Herpesviridae': 'firebrick',
}

VIRUSES = [
    # Herpesvirus
    'Herpesviridae, Herpes Simplex 1, KOS',
    'Herpesviridae, Herpes Simplex 2, Strain G',
    'Herpesviridae, Varicella Zoster Virus, Ellen Strain',
    'Herpesviridae, Epstein Barr Virus',
    'Herpesviridae, Human cytomegalovirus',
    'Herpesviridae, 6B',
    'Herpesviridae, Human Herpes 7',
    'Herpesviridae, Kaposi Sarcoma (HHV-8)',
    # Adenovirus
    'Adenoviridae, Type 1, Strain Adenoid 71',
    'Adenoviridae, Type 3, Strain GB',
    'Adenoviridae, Type 4, Strain RI-67',
    'Adenoviridae, Type 5, Strain Adenoid 75',
    'Adenoviridae, Type 7, Strain Gomen',
    'Adenoviridae, Type 11, Strain Slobitski',
    'Adenoviridae, Type 14, Strain de Wit',
    'Adenoviridae, Type 37, Strain GW (76-19026)',
    # Papillomavirus
    'Papillomaviridae, Type 1',
    'Papillomaviridae, Type 2',
    'Papillomaviridae, Type 5',
    'Papillomaviridae, Type 6b',
    'Papillomaviridae, Type 11',
    'Papillomaviridae, Type 16',
    'Papillomaviridae, Type 18',
    'Papillomaviridae, Type 52',
    # Polyomavirus
    'Polyomaviridae, BK, Strain MM',
    'Polyomaviridae, JC, Strain MAD-4',
    'Polyomaviridae, Merkel Cell Polyoma, Strain MKL-1',
]

GENE_KINETIC_GROUPS = ['IE', 'E', 'E/L', 'L', 'Latent']

DNA_BASES = ['A', 'C', 'G', 'T']
DNA_COMPLEMENT_MAP = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

MPRA_FLANK_UPSTREAM = (
    'ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAG'
    'GTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTC'
    'CATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG'
)

MPRA_FLANK_DOWNSTREAM = (
    'CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTA'
    'TATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATC'
    'CTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT'
)

# constants used for motif annotation
TACS_WINDOW = 5
PER_POS_THRESHOLD = 0.15
PEAK_RADIUS = 5
OVERLAP_THRESHOLD = 0.2

TF_GENES_K562 = [
    'SP/KLF', 'ELF', 'GATA', 'FOS:JUN', 'CTCF', 'NFY', 'USF', 'CREB', 'NRF1',
    'ZBTB33', 'ATF2', 'ZNF460', 'ZNF143', 'YY1', 'E2F', 'STAT', 'ELK:SREBF2',
    'ELK', 'CTCFL', 'POU', 'IRF3', 'MAZ', 'BATF3', 'SRF', 'GFI1B', 'NFIB'
]

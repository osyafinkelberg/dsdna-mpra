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

CELL_LINES = ['GM12878', 'Jurkat', 'MRC5', 'A549', 'HEK293', 'K562']
CELL_LINES_COLORS = ['navy', 'cornflowerblue', 'forestgreen', 'deeppink', 'orangered', 'firebrick']

DSDNA_FAMILIES = ['Polyomaviridae', 'Papillomaviridae', 'Adenoviridae', 'Herpesviridae']
VIRUSES = [
    # Polyomavirus
    'Polyomaviridae, BK, Strain MM',
    'Polyomaviridae, JC, Strain MAD-4',
    'Polyomaviridae, Merkel Cell Polyoma, Strain MKL-1',
    # Papillomavirus
    'Papillomaviridae, Type 1',
    'Papillomaviridae, Type 2',
    'Papillomaviridae, Type 5',
    'Papillomaviridae, Type 6b',
    'Papillomaviridae, Type 11',
    'Papillomaviridae, Type 16',
    'Papillomaviridae, Type 18',
    'Papillomaviridae, Type 52'
    # Adenovirus
    'Adenoviridae, Type 1, Strain Adenoid 71',
    'Adenoviridae, Type 3, Strain GB',
    'Adenoviridae, Type 4, Strain RI-67',
    'Adenoviridae, Type 5, Strain Adenoid 75',
    'Adenoviridae, Type 7, Strain Gomen',
    'Adenoviridae, Type 11, Strain Slobitski',
    'Adenoviridae, Type 14, Strain de Wit',
    'Adenoviridae, Type 37, Strain GW (76-19026)',
    # Herpesvirus
    'Herpesviridae, Herpes Simplex 1, KOS',
    'Herpesviridae, Herpes Simplex 2, Strain G',
    'Herpesviridae, Varicella Zoster Virus, Ellen Strain',
    'Herpesviridae, Epstein Barr Virus',
    'Herpesviridae, Human cytomegalovirus',
    'Herpesviridae, 6B',
    'Herpesviridae, Human Herpes 7',
    'Herpesviridae, Kaposi Sarcoma (HHV-8)',
]

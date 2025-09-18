# dsDNA-MPRA: Analysis and Toolkit for Double-Stranded DNA viruses MPRA Data

---

## Project Overview

This repository contains the code and software developed for the manuscript:

**Tommy H. Taslim, Joseph A. Finkelberg, et al.** (2025).  
*Global cis-regulatory landscape of double-stranded DNA viruses*.  
[bioRxiv: 10.1101/2025.07.20.665756v1](https://www.biorxiv.org/content/10.1101/2025.07.20.665756v1)

The repository includes the `dsdna-mpra` Python package for processing and analyzing MPRA data from double-stranded DNA viruses.

---

## Installation

To install locally:

```bash
git clone https://github.com/osyafinkelberg/dsdna-mpra.git
cd dsdna-mpra

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

### Configuration

By default, the base project directory (`BASE_DIR`) is automatically inferred from the code structure.

If needed, you can override it by setting the environment variable:

```bash
export DSDNA_MPRA_BASE_DIR=/your/path/to/dsdna-mpra
```

### External Tools

By default, the project expects HOMER to be installed at:

```bash
<BASE_DIR>/../soft/homer
```

If HOMER is installed elsewhere, set the environment variable:

```bash
export PATH_TO_HOMER=/custom/path/to/homer
```

### Input Data

The raw MPRA data was preprocessed using the published MPRA processing pipeline: [MPRAsuite](https://github.com/tewhey-lab/MPRASuite) by Tewhey Lab. The output of this preprocessing (quantified barcode counts and activity scores) is available in the Zenodo repository and serves as the starting point for all downstream analyses in this project.

Preprocessed input data required to reproduce the analyses is available at:  
https://zenodo.org/record/16379986  
(DOI: [10.5281/zenodo.16379986](https://doi.org/10.5281/zenodo.16379986))

Download and unzip the archive so that the contents are placed under the `data/raw/` directory within the project root:

```bash
# From within the project root
wget https://zenodo.org/record/16379986/files/dsdna-mpra-data.zip
unzip dsdna-mpra-data.zip
```

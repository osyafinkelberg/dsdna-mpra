import sys

import pandas as pd
from Bio import Entrez, SeqIO

sys.path.insert(0, '..')
from dsdna_mpra import config  # noqa E402


def extract_cds_positions(genome_id: str) -> pd.DataFrame:
    cds_data = []
    with Entrez.efetch(db="nucleotide", id=genome_id, rettype="gb", retmode="text") as handle:
        record = SeqIO.read(handle, "genbank")
        for feature in record.features:
            if feature.type == "CDS":
                gene_name = feature.qualifiers.get("gene", [""])[0]
                protein_name = feature.qualifiers.get("product", [""])[0]
                strand = '+' if feature.location.strand == 1 else '-'
                begin = int(feature.location.start)
                end = int(feature.location.end)
                cds_data.append({
                    "genome": genome_id,
                    "gene_name": gene_name,
                    "protein_name": protein_name,
                    "strand": strand,
                    "begin": begin,
                    "end": end
                })
    return pd.DataFrame(cds_data)


def extract_gene_positions(genome_id: str) -> pd.DataFrame:
    records = []
    with Entrez.efetch(db="nucleotide", id=genome_id, rettype="gb", retmode="text") as handle:
        record = SeqIO.read(handle, "genbank")
        for feature in record.features:
            if feature.type == "gene":
                gene_name = feature.qualifiers.get("gene", [""])[0]
                strand = '+' if feature.location.strand == 1 else '-'
                start = int(feature.location.start)
                end = int(feature.location.end)
                five_prime = start if strand == '+' else end
                three_prime = end if strand == '+' else start
                records.append({
                    "genome": genome_id,
                    "gene_name": gene_name,
                    "strand": strand,
                    "five_prime": five_prime,
                    "three_prime": three_prime
                })
    return pd.DataFrame(records)


def main() -> None:
    virus_genomes = pd.read_csv(config.RAW_DIR / 'virus_genbank_ids.txt').columns.values
    Entrez.email = config.ENTREZ_EMAIL

    # collect and save CDS positions
    cds_df = pd.concat([extract_cds_positions(genome_id) for genome_id in virus_genomes]).reset_index(drop=True)
    cds_df.to_csv(config.PROCESSED_DIR / 'virus_cds_positions.csv', index=False)

    # collect and save gene start positions
    gene_df = pd.concat([extract_gene_positions(genome_id) for genome_id in virus_genomes]).reset_index(drop=True)
    gene_df.to_csv(config.PROCESSED_DIR / 'virus_gene_positions.csv', index=False)


if __name__ == "__main__":
    main()

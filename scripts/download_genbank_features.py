import sys

import numpy as np
import pandas as pd
from Bio import Entrez, SeqIO
from Bio.SeqFeature import CompoundLocation

sys.path.insert(0, '..')
from dsdna_mpra import config, clustering  # noqa E402


def extract_cds_positions(genome_id: str) -> pd.DataFrame:
    cds_data = []
    with Entrez.efetch(db="nucleotide", id=genome_id, rettype="gb", retmode="text") as handle:
        record = SeqIO.read(handle, "genbank")
        for feature in record.features:
            if feature.type == "CDS":
                gene_name = feature.qualifiers.get("gene", [""])[0]
                protein_name = feature.qualifiers.get("product", [""])[0]
                strand = '+' if feature.location.strand == 1 else '-'
                if isinstance(feature.location, CompoundLocation):
                    for part in feature.location.parts:
                        cds_data.append({
                            "genome": genome_id,
                            "gene_name": gene_name,
                            "protein_name": protein_name,
                            "strand": strand,
                            "begin": int(part.start),
                            "end": int(part.end)
                        })
                else:
                    cds_data.append({
                        "genome": genome_id,
                        "gene_name": gene_name,
                        "protein_name": protein_name,
                        "strand": strand,
                        "begin": int(feature.location.start),
                        "end": int(feature.location.end)
                    })
    return pd.DataFrame(cds_data)


def extract_gene_positions(genome_id: str) -> pd.DataFrame:
    records = []
    with Entrez.efetch(db="nucleotide", id=genome_id, rettype="gb", retmode="text") as handle:
        record = SeqIO.read(handle, "genbank")
        features = [f for f in record.features if f.type == "gene"]
        if not features:
            features = [f for f in record.features if f.type == "mRNA"]
        for feature in features:
            if feature.type == "gene":
                gene_name = feature.qualifiers.get("gene", [""])[0]
            else:  # mRNA fallback
                gene_name = feature.qualifiers.get(
                    "gene", feature.qualifiers.get("product",  feature.qualifiers.get("transcript_id", [""]))
                )[0]
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


def extract_intergenic_regions(
    gene_df: pd.DataFrame, cds_df: pd.DataFrame, genome_sizes: dict[str, int]
) -> pd.DataFrame:
    intergenic_regions = []
    for genome, genome_size in genome_sizes.items():
        genes = gene_df[gene_df.genome == genome]
        cdss = cds_df[cds_df.genome == genome]
        merged_genes = clustering.merge(np.sort(genes[['five_prime', 'three_prime']].values, axis=1))
        merged_cdss = clustering.merge(np.sort(cdss[['begin', 'end']].values, axis=1))
        merged_regions = clustering.merge(np.concatenate([merged_genes, merged_cdss]))
        prev_end = 0
        for start, end in merged_regions:
            if prev_end < start:
                intergenic_regions.append({
                    "genome": genome,
                    "begin": prev_end,
                    "end": start,
                })
            prev_end = end
        if prev_end < genome_size:
            intergenic_regions.append({
                "genome": genome,
                "begin": prev_end,
                "end": genome_size,
            })
    return pd.DataFrame(intergenic_regions)


def main() -> None:
    virus_genomes = pd.read_csv(config.RAW_DIR / 'virus_genbank_ids.txt').columns.values.tolist()
    virus_genomes += ['V01555.2', 'BK012101.1', 'GQ994935.1', 'NC_001348.1']  # CAGE-seq genome records
    Entrez.email = config.ENTREZ_EMAIL

    # collect and save CDS positions
    cds_df = pd.concat([extract_cds_positions(genome_id) for genome_id in virus_genomes]).reset_index(drop=True)
    cds_df.to_csv(config.PROCESSED_DIR / 'virus_cds_positions.csv', index=False)

    # collect and save gene start positions
    gene_df = pd.concat([extract_gene_positions(genome_id) for genome_id in virus_genomes]).reset_index(drop=True)
    gene_df.to_csv(config.PROCESSED_DIR / 'virus_gene_positions.csv', index=False)

    # infer intergenic regions from gene body positions
    genome_sizes = dict(
        pd.read_csv(
            config.PROCESSED_DIR / 'summary_virus_genome_records.csv'
        )[['accession_id', 'genome_size']].values
    )
    intergenic_df = extract_intergenic_regions(gene_df, cds_df, genome_sizes)
    intergenic_df.to_csv(config.PROCESSED_DIR / 'virus_intergenic_positions.csv', index=False)


if __name__ == "__main__":
    main()

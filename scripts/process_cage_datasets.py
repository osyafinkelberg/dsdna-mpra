import sys
import pandas as pd

from Bio import Entrez, SeqIO

sys.path.insert(0, '..')
from dsdna_mpra import config, clustering  # noqa E402


def main() -> None:
    # HHV-1
    Entrez.email = config.ENTREZ_EMAIL
    with Entrez.efetch(
        db="nucleotide", id="BK012101.1", rettype="gb", retmode="text"
    ) as handle:
        record = SeqIO.read(handle, "genbank")

    rows = []
    for feature in record.features:
        if feature.type.lower() == "mrna":
            strand = feature.location.strand
            start = int(feature.location.start)
            end = int(feature.location.end)

            if strand == 1:
                five_prime = start
                strand_symbol = '+'
            else:
                five_prime = end
                strand_symbol = '-'

            rows.append({
                "genome": 'BK012101.1',
                "strand": strand_symbol,
                "five_prime": five_prime,
            })

    output_path = config.PROCESSED_DIR / 'cage_pmid_32341360_gbid_BK012101.1.csv'
    pd.DataFrame(rows).to_csv(output_path, index=False)

    # HHV-3
    raw_df = pd.read_excel(
        config.RAW_DIR / 'pmid_33024035_table_s1.xlsx',
        skiprows=3
    ).iloc[:, 1:10]

    raw_df.columns = [
        'pos', 'strand', 'drna_position', 'drna_value', 'drna_difference',
        'cage_position', 'cage_value', 'cage_note', 'annotation'
    ]

    raw_df = raw_df[
        raw_df.cage_note.isna() & raw_df.cage_position.notna()
    ].reset_index(drop=True)

    output_path = config.PROCESSED_DIR / 'cage_pmid_33024035_gbid_NC_001348.1.csv'
    pd.DataFrame({
        'genome': 'NC_001348.1',
        'strand': raw_df.strand.values,
        'five_prime': raw_df.cage_position.values.astype(int),
    }).to_csv(output_path, index=False)

    # HHV-4
    raw_df = pd.read_excel(config.RAW_DIR / 'pmid_29864140_table_s1.xls')

    raw_df = raw_df[
        (raw_df['ORF/promoter*'] != 'artifact') &
        ~(raw_df.cluster_strand.isna())
    ].reset_index(drop=True)

    five_prime_mean = raw_df[['cluster_start', 'cluster_end']].values.mean(axis=1).astype(int)
    pd.DataFrame({
        'genome': 'V01555.2',
        'strand': raw_df.cluster_strand.values,
        'five_prime': five_prime_mean,
    }).to_csv(config.PROCESSED_DIR / 'cage_pmid_29864140_gbid_V01555.2.csv', index=False)

    # gene expression kinetics
    pd.DataFrame({
        'genome': 'V01555.2',
        'strand': raw_df.cluster_strand.values,
        'five_prime': five_prime_mean,
        'kinetics': raw_df.Kinetics,
    }).to_csv(config.PROCESSED_DIR / 'cage_pmid_29864140_gbid_V01555.2_kinetics.csv', index=False)

    # HHV-8
    raw_df = pd.read_excel(
        config.RAW_DIR / 'pmid_38206015_supplemental_tables.xlsx',
        sheet_name='Supplemental Table 1A - TSS',
        skiprows=4
    ).iloc[:-7]

    raw_df = raw_df[
        (raw_df['CAGE score'] > 0) & (raw_df['RAMPAGE'] == 'y')
    ].sort_values('Position').reset_index(drop=True)

    output_path = config.PROCESSED_DIR / 'cage_pmid_38206015_gbid_GQ994935.1.csv'
    pd.DataFrame({
        'genome': 'GQ994935.1',
        'strand': raw_df.strand.values,
        'five_prime': raw_df.Position.values,
    }).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

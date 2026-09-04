"""Nhan HLA: sniff dau phan cach, chuan hoa gia tri allele ve dang khong tien to gene."""
import pandas as pd


def normalize_allele(value) -> str:
    """'A*33:03' -> '33:03'; '' va NaN -> ''.

    Nhan HAN ghi "A*33:03", du doan chi mang "33:03". Lech tien to nay khong
    no, no lam F1 = 0 im lang neu bo qua.
    """
    if not isinstance(value, str):
        return ""
    value = value.strip()
    return value.split("*", 1)[1] if "*" in value else value


def load_labels(path: str, genes: list[str], n_digits: int) -> pd.DataFrame:
    """Sniff dau phan cach (tab hay phay). Cot <GENE>_1/<GENE>_2, da normalize."""
    with open(path, "r") as fh:
        delimiter = "\t" if "\t" in fh.readline() else ","
    columns = [gene.upper() + "_" + x for gene in genes for x in ("1", "2")]
    df = pd.read_csv(path, sep=delimiter, index_col=0, dtype=str)[columns].copy()
    # dtype=str doesn't reach the index column -- HAN's sample_id is bare
    # integers, so without this df.index is int64 and every downstream
    # '<id>_1'/'<id>_2' string build (dataset.py) dies with a TypeError.
    df.index = df.index.astype(str)
    for col in columns:
        df[col] = df[col].apply(normalize_allele)
        if n_digits is not None:
            df[col] = df[col].apply(
                lambda x: ":".join(x.split(":")[:n_digits // 2]).split("/")[0])
    return df

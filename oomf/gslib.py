import pandas as pd



def read(filename: str, nan_value=-999.0) -> pd.DataFrame:
    with open(filename, "r") as f:
        lines = f.readlines()
        ncols = int(lines[1].split()[0])
        col_names = [lines[i + 2].strip() for i in range(ncols)]
    df = pd.read_csv(
        filename,
        skiprows=ncols + 2,
        delim_whitespace=True,
        names=col_names,
        na_values=nan_value,
    )
    return df


def write(df: pd.DataFrame, filename: str) -> None:
    with open(filename,"w") as f:
        f.write("GSLIB Example Data\n")
        f.write(f"{len(df.columns)}\n")
        f.write("\n".join(df.columns) + "\n")
        for row in df.itertuples():
            row_data = "\t".join([f"{i:.3f}" for i in row[1:]])
            f.write(f"{row_data}\n")
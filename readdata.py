import pandas as pd

i=0
print("start")
for df in pd.read_csv(r"data/scp_gex_matrix.csv.gz",compression ="gzip", chunksize=100 ):
    print(i)
    i = i + 1
    print(df)


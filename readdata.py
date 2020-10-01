import pandas as pd
import numpy as np

print("start")
z = np.empty((0,122558))
for df in pd.read_csv(r"data/scp_gex_matrix.csv.gz",compression ="gzip", chunksize=50):
    df = df[(df.T != 0).any()]
    print(df)
    arr = df.to_numpy()
    z = np.vstack([z,arr])
    print(z)



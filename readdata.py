import pandas as pd
import numpy as np

print("start")
z = np.empty((0,122558)) # Will have memory error soon
for df in pd.read_csv(r"data/scp_gex_matrix.csv.gz",compression ="gzip", chunksize=10):
    df = df[(df.loc[:, df.columns != "GENE"].T != 0).any()] # remove all zero rows
    print(df)
    arr = df.to_numpy()
    z = np.vstack([z,arr]) # error after 450 rows 
    print(z)

idx = np.argwhere(np.all(z[..., :] ==0,axis = 0)) #find index of zero columns
z = np.delete(z,idx,axis = 1) #remove all zero columns

#continue processing on z

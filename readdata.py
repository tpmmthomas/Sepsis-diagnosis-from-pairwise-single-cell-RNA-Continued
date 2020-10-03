import pandas as pd
import numpy as np
import gzip
import csv 

filename = r"data/scp_gex_matrix.csv.gz"
print("start")
with gzip.open(filename, mode="rt") as f:
    reader = csv.reader(f)
    i = next(reader) #stores name of each sample

col_name = np.array(i)
print(col_name)
data = np.empty((0,122558)) 

for df in pd.read_csv(filename,compression ="gzip", chunksize=1):
    df = df[(df.loc[:, df.columns != "GENE"].T != 0).any()] # remove all zero rows
    print(df)
    arr = df.to_numpy()
    data = np.vstack([data,arr]) 
    print(data)

idx = np.argwhere(np.all(data[..., :] ==0,axis = 0)) #find index of zero columns
z = np.delete(z,idx,axis = 1) #remove all zero columns

#continue processing on z

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 
from statsmodels.stats.multitest import fdrcorrection
from sklearn.model_selection import train_test_split
import gzip
import csv 

def fprint(txtt):
    f = open(r"/uac/cprj/cprj2716/tryval.txt","a+")
    f.write(str(txtt))
    f.write("\n")
    f.close()

fprint("start") 

xtrain = r"/research/dept8/estr3108/cprj2716/training_sample.csv.gz"
ytrain = r"/research/dept8/estr3108/cprj2716/training_label.csv.gz"
xtest = r"/research/dept8/estr3108/cprj2716/testing_sample.csv.gz"
ytest = r"/research/dept8/estr3108/cprj2716/testing_label.csv.gz"
rnaname = r"/research/dept8/estr3108/cprj2716/rna_name.csv.gz"
fdrn = r"/research/dept8/estr3108/cprj2716/P_fdr2.csv.gz"


samplesdf = pd.DataFrame()
for df in  pd.read_csv(xtrain,compression ="gzip",delimiter=',', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
x_train = samplesdf.to_numpy()
fprint("Read xtrain")
samplesdf = pd.DataFrame()
for df in  pd.read_csv(ytrain,compression ="gzip",delimiter=',', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
y_train = samplesdf.to_numpy()
fprint("Read ytrain")
samplesdf = pd.DataFrame()
for df in  pd.read_csv(xtest,compression ="gzip",delimiter=',', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
x_test = samplesdf.to_numpy()
fprint("Read xtest")
samplesdf = pd.DataFrame()
for df in  pd.read_csv(ytest,compression ="gzip",delimiter=',', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
y_test = samplesdf.to_numpy()
fprint("Read ytest")
samplesdf = pd.DataFrame()
for df in  pd.read_csv(rnaname,compression ="gzip",delimiter=',', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
rna_name = samplesdf.to_numpy()
fprint("Read rnaname")
samplesdf = pd.DataFrame()
for df in  pd.read_csv(fdrn,compression ="gzip",delimiter=',', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
P_fdr = samplesdf.to_numpy()
fprint("Read fdr")

#checking again
fprint(x_train.shape)
fprint(x_test.shape)
fprint(y_train.shape)
fprint(y_test.shape)
fprint(rna_name.shape)
fprint(P_fdr.shape)


# select RNAs with small P_fdr as the input
i = 0
idx = []
for fdr in P_fdr:
    if fdr > 0.005:
        idx.append(i)
    i = i + 1

fprint("Filter succcessful")
x_train = np.delete(x_train,idx,1)
x_test = np.delete(x_test,idx,1)
rna_name = np.delete(rna_name,idx)

# Checking
fprint(x_train.shape)
fprint(x_test.shape)
fprint(y_train.shape)
fprint(y_test.shape)
fprint(rna_name.shape)


# Save all files
df = pd.DataFrame(P_fdr)
df.to_csv(r"/research/dept8/estr3108/cprj2716/P_fdr3.csv.gz",index=False,sep=",",compression="gzip")
fprint("Saved1")
df = pd.DataFrame(x_train)
df.to_csv(r"/research/dept8/estr3108/cprj2716/training_sample3.csv.gz",index=False,sep=",",compression="gzip")
fprint("Saved2")
df = pd.DataFrame(x_test)
df.to_csv(r"/research/dept8/estr3108/cprj2716/testing_sample3.csv.gz",index=False,sep=",",compression="gzip")
fprint("Saved3")
df = pd.DataFrame(y_train)
df.to_csv(r"/research/dept8/estr3108/cprj2716/training_label3.csv.gz",index=False,sep=",",compression="gzip")
fprint("Saved4")
df = pd.DataFrame(y_test)
df.to_csv(r"/research/dept8/estr3108/cprj2716/testing_label3.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(rna_name)
fprint("Saved5")
df.to_csv(r"/research/dept8/estr3108/cprj2716/rna_name3.csv.gz",index=False,sep=",",compression="gzip")
fprint("Saved6")

# Checking
fprint("Finish!")


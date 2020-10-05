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
    f = open(r"/uac/cprj/cprj2716/dp2.txt","a+")
    f.write(str(txtt))
    f.write("\n")
    f.close()


fprint("start")
xtrain = r"/research/dept8/estr3108/cprj2716/training_sample_raw.csv.gz"
ytrain = r"/research/dept8/estr3108/cprj2716/training_label_raw.csv.gz"
xtest = r"/research/dept8/estr3108/cprj2716/testing_sample_raw.csv.gz"
ytest = r"/research/dept8/estr3108/cprj2716/testing_label_raw.csv.gz"
rnaname = r"/research/dept8/estr3108/cprj2716/rna_names_raw.csv.gz"

samplesdf = pd.DataFrame()
for df in  pd.read_csv(xtrain,compression ="gzip",delimiter=' ', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
x_train = samplesdf.to_numpy()

samplesdf = pd.DataFrame()
for df in  pd.read_csv(xtest,compression ="gzip",delimiter=' ', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
x_test = samplesdf.to_numpy()

samplesdf = pd.DataFrame()
for df in  pd.read_csv(ytest,compression ="gzip",delimiter=' ', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
y_test = samplesdf.to_numpy()

samplesdf = pd.DataFrame()
for df in  pd.read_csv(ytrain,compression ="gzip",delimiter=' ', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
y_train = samplesdf.to_numpy()

samplesdf = pd.DataFrame()
for df in  pd.read_csv(rnaname,compression ="gzip",delimiter=' ', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
rna_name = samplesdf.to_numpy()


#checking 
fprint(x_train.shape)
fprint(x_test.shape)
fprint(y_train.shape)
fprint(y_test.shape)
fprint(rna_name.shape)

#separate control and case samples
i = 0
idx = []
for lb in y_train:
    fprint(lb)
    if lb == "Control":
        idx.append(i)
    i = i + 1

fprint("Now mask.")
mask = np.ones(len(x_train), dtype=bool)
mask[idx,] = False
con_sample,case_sample = x_train[idx], x_train[mask]


fprint("Separation succcessful")
fprint(con_sample.shape)
fprint(case_sample.shape)

df = pd.DataFrame(con_sample)
df.to_csv(r"/research/dept8/estr3108/cprj2716/con_sample.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(case_sample)
df.to_csv(r"/research/dept8/estr3108/cprj2716/case_sample.csv.gz",index=False,sep=",",compression="gzip")

# optain stat value
t_stat,pvalue = stats.ttest_ind(con_sample, case_sample, axis = 0, equal_var=True, nan_policy='raise')
rejected, P_fdr = fdrcorrection(pvalue, alpha=0.05, method='indep', is_sorted=False)

fprint("Value computed!")
fprint(P_fdr.shape)

# select RNAs with small P_fdr as the input
i = 0
for fdr in P_fdr:
    if fdr > 0.05:
        x_train = np.delete(x_train,i,1)
        x_test = np.delete(x_test,i,1)
        rna_names = np.delete(rna_names,i)
        i = i + 1

fprint("Filter succcessful")

# Save all files
df = pd.DataFrame(P_fdr)
df.to_csv(r"/research/dept8/estr3108/cprj2716/P_fdr.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(x_train)
df.to_csv(r"/research/dept8/estr3108/cprj2716/training_sample.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(x_test)
df.to_csv(r"/research/dept8/estr3108/cprj2716/testing_sample.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(y_train)
df.to_csv(r"/research/dept8/estr3108/cprj2716/training_label.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(y_test)
df.to_csv(r"/research/dept8/estr3108/cprj2716/testing_label.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(rna_names)
df.to_csv(r"/research/dept8/estr3108/cprj2716/rna_name.csv.gz",index=False,sep=",",compression="gzip")

# Checking
fprint(x_train.shape)
fprint(x_test.shape)
fprint(y_train.shape)
fprint(y_test.shape)
fprint(rna_names.shape)
fprint("Finish!")


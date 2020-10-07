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
    f = open(r"/uac/cprj/cprj2716/dp4.txt","a+")
    f.write(str(txtt))
    f.write("\n")
    f.close()


fprint("start")
xtrain = r"/research/dept8/estr3108/cprj2716/training_sample.csv.gz"
ytrain = r"/research/dept8/estr3108/cprj2716/training_label.csv.gz"
xtest = r"/research/dept8/estr3108/cprj2716/testing_sample.csv.gz"
ytest = r"/research/dept8/estr3108/cprj2716/testing_label.csv.gz"
rnaname = r"/research/dept8/estr3108/cprj2716/rna_names_raw.csv.gz"
con = r"/research/dept8/estr3108/cprj2716/con_sample.csv.gz"
case = r"/research/dept8/estr3108/cprj2716/case_sample.csv.gz"

samplesdf = pd.DataFrame()
for df in  pd.read_csv(xtrain,compression ="gzip",delimiter=',', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
x_train = samplesdf.to_numpy()

samplesdf = pd.DataFrame()
for df in  pd.read_csv(ytrain,compression ="gzip",delimiter=',', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
y_train = samplesdf.to_numpy()

samplesdf = pd.DataFrame()
for df in  pd.read_csv(xtest,compression ="gzip",delimiter=',', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
x_test = samplesdf.to_numpy()

samplesdf = pd.DataFrame()
for df in  pd.read_csv(ytest,compression ="gzip",delimiter=',', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
y_test = samplesdf.to_numpy()

samplesdf = pd.DataFrame()
for df in  pd.read_csv(rnaname,compression ="gzip",delimiter=' ', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
rna_name = samplesdf.to_numpy()

samplesdf = pd.DataFrame()
for df in  pd.read_csv(con,compression ="gzip",delimiter=',', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
con_sample = samplesdf.to_numpy()

samplesdf = pd.DataFrame()
for df in  pd.read_csv(case,compression ="gzip",delimiter=',', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
case_sample = samplesdf.to_numpy()

fprint(x_train.shape)
fprint(x_test.shape)
fprint(y_train.shape)
fprint(y_test.shape)
fprint(rna_name.shape)
fprint(con_sample.shape)
fprint(case_sample.shape)

#to avoid error, delete zero columns of case samples
idx = np.argwhere(np.all(case_sample==0,axis = 0)) #find index of zero columns
idx2 = np.argwhere(np.all(con_sample==0,axis = 0))
idx = np.intersect1d(idx,idx2)
case_sample = np.delete(case_sample,idx,axis = 1) 
con_sample = np.delete(con_sample,idx,axis = 1) 
x_train = np.delete(x_train,idx,axis = 1) 
x_test = np.delete(x_test,idx,axis = 1) 
rna_name = np.delete(rna_name,idx) 

fprint("After deletion")
fprint(x_train.shape)
fprint(x_test.shape)
fprint(y_train.shape)
fprint(y_test.shape)
fprint(rna_name.shape)
fprint(con_sample.shape)
fprint(case_sample.shape)

# optain stat value
t_stat,pvalue = stats.ttest_ind(con_sample, case_sample, axis = 0, equal_var=True, nan_policy='raise')
rejected, P_fdr = fdrcorrection(pvalue, alpha=0.05, method='indep', is_sorted=False)

fprint("Value computed!")
fprint(P_fdr.shape)

# select RNAs with small P_fdr as the input
i = 0
idx = []
for fdr in P_fdr:
    fprint(fdr)
    if fdr > 0.1:
        idx.append(i)
    i = i + 1

x_train = np.delete(x_train,idx,1)
x_test = np.delete(x_test,idx,1)
rna_name = np.delete(rna_name,idx)


fprint("Filter succcessful")

# Checking
fprint(x_train.shape)
fprint(x_test.shape)
fprint(y_train.shape)
fprint(y_test.shape)
fprint(rna_name.shape)


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
df = pd.DataFrame(rna_name)
df.to_csv(r"/research/dept8/estr3108/cprj2716/rna_name.csv.gz",index=False,sep=",",compression="gzip")

# Checking

fprint("Finish!")


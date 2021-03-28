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
# org all 2. When need 0.1 remember change back.  This is 0.005.
xtrain = r"/research/dept8/estr3108/cprj2716/training_sample3.csv.gz"
ytrain = r"/research/dept8/estr3108/cprj2716/training_label2_int2.csv.gz"
xtest = r"/research/dept8/estr3108/cprj2716/testing_sample3.csv.gz"
ytest = r"/research/dept8/estr3108/cprj2716/testing_label2_int2.csv.gz"
rnaname = r"/research/dept8/estr3108/cprj2716/rna_name3.csv.gz"


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

#checking again
fprint(x_train.shape)
fprint(x_test.shape)
fprint(y_train.shape)
fprint(y_test.shape)
fprint(rna_name.shape)

# Delete sparse sample

idx = []
i = 0
for samples in x_train :
    num0 = 0
    for col in samples:
        if col == 0:
            num0 = num0 + 1
    if num0/len(samples)>0.9:
        idx.append(i)
    i = i + 1

x_train = np.delete(x_train, idx,axis=0)
y_train = np.delete(y_train,idx,axis=0)

fprint("Deleted x_train sparse.")
fprint(x_train.shape)
fprint(y_train.shape)

# Delete sparse sample

idx = []
i = 0
for samples in x_test :
    num0 = 0
    for col in samples:
        if col == 0:
            num0 = num0 + 1
    if num0/len(samples)>0.9:
        idx.append(i)
    i = i + 1

x_test = np.delete(x_test, idx,axis=0)
y_test = np.delete(y_test, idx,axis=0)

fprint("Deleted x_test sparse.")
fprint(x_test.shape)
fprint(y_test.shape)

#Take transpose of everything
allsample = np.vstack((x_train,x_test))
allsample = np.transpose(allsample)
fprint("Taken transpose.")
fprint(allsample.shape)


# Delete sparse sample

idx = []
i = 0
for rna in allsample :
    num0 = 0
    for col in rna:
        if col == 0:
            num0 = num0 + 1
    if num0/len(rna)>0.9:
        idx.append(i)
    i = i + 1

fprint("Number of cols found")
fprint(len(idx))
x_train = np.delete(x_train,idx,axis=1)
x_test = np.delete(x_test,idx,axis=1)
rna_name = np.delete(rna_name,idx,axis=0)

fprint("Deleted col sparse.")
fprint(x_train.shape)
fprint(x_test.shape)
fprint(rna_name.shape)


# Save all files

df = pd.DataFrame(x_train)
df.to_csv(r"/research/dept8/estr3108/cprj2716/training_sample_3s.csv.gz",index=False,sep=",",compression="gzip")
fprint("Saved2")
df = pd.DataFrame(x_test)
df.to_csv(r"/research/dept8/estr3108/cprj2716/testing_sample2_3s.csv.gz",index=False,sep=",",compression="gzip")
fprint("Saved3")
df = pd.DataFrame(y_train)
df.to_csv(r"/research/dept8/estr3108/cprj2716/training_label_3s.csv.gz",index=False,sep=",",compression="gzip")
fprint("Saved4")
df = pd.DataFrame(y_test)
df.to_csv(r"/research/dept8/estr3108/cprj2716/testing_label_3s.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(rna_name)
fprint("Saved5")
df.to_csv(r"/research/dept8/estr3108/cprj2716/rna_name_3s.csv.gz",index=False,sep=",",compression="gzip")
fprint("Saved6")

# Checking
fprint("Finish!")


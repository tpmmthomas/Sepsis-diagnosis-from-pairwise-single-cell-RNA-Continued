import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 
from statsmodelsSelf  import fdrcorrection
from sklearn.model_selection import train_test_split
import gzip
import csv 

sample_f = r"scp_gex_matrix.csv.gz" 
label_f = r"scp_meta.txt"

def fprint(txtt):
    f = open(r"dp.txt","a+")
    f.write(str(txtt))
    f.write("\n")
    f.close()

fprint("start123") 

#read data
samplesdf = pd.DataFrame()
for df in  pd.read_csv(sample_f,compression ="gzip", chunksize = 1000, header = 0):
    fprint("check")
    samplesdf = samplesdf.append(df)
samplesdf = samplesdf.T
samples_name = samplesdf.index.values
samples_name = samples_name[1:]
samples = samplesdf.values
samples = samples[1:]
fprint("First!")
labels_name = np.array([])
labels_temp = np.array([])
i = 0
with open(label_f) as rawlabel:
    label_reader = csv.reader(rawlabel, delimiter='\t')
    for labels in label_reader:
        if i>=2:
            labels_name = np.append(labels_name,[labels[0]])
            labels_temp = np.append(labels_temp,[labels[3]])
        i = i + 1
fprint("Second!")
labels_text = np.array([])
#find correct labels for each element in samples
for lb in samples_name:
    idx = np.where(labels_name == lb)
    if len(idx[0]) > 1:
        fprint("Warning! Have duplicate.")
    elif len(idx[0]) == 0:
        fprint("Warning! No corresponding entry.")
    labels_text = np.append(labels_text, labels_temp[idx[0][0]])
    
fprint("Samples ")
fprint(samples.shape)
fprint("Labels ")
fprint(labels_text.shape)
fprint(labels_text[0])

#convert labels_text into corresponding labels(0: non-sepsis, 1: sepsis)
labels = np.array([])
num1=0
num0=0
for lb in labels_text:
    if lb == "Int-URO" or lb == "URO" or lb == "Bac-SEP" or lb == "ICU-SEP":
        labels = np.append(labels,[1])
        num1 = num1 + 1
    else:
        labels = np.append(labels,[0])
        num0 = num0 + 1

fprint("Converted to label in numbers")
fprint(labels.shape)
fprint(num1)
fprint(num0)


#delete all zero columns and rows
idx = np.argwhere(np.all(samples==0,axis = 0)) #find index of zero columns
samples = np.delete(samples,idx,axis = 1) 

fprint("After deletion of columns:")
fprint(samples.shape)

idx = np.where(np.all(np.isclose(samples,0),axis=1))
samples = np.delete(samples,idx,axis = 0)
labels = np.delete(labels,idx)

fprint("After deletion of rows:")
fprint(samples.shape)

fprint("Labels check")
fprint(labels.shape)

#split training and testing sample (x = sample, y = label)
x_train,x_test,y_train,y_test = train_test_split(samples,labels,test_size = 0.1, random_state = 5)
fprint("Split succcessful")

#checking 
fprint(x_train.shape)
fprint(x_test.shape)
fprint(y_train.shape)
fprint(y_test.shape)

#separate control and case samples
i = 0
idx = []
for lb in y_train:
    fprint(lb)
    if lb == 0:
        idx.append(i)
    i = i + 1

fprint("Now mask.")
mask = np.ones(len(x_train), dtype=bool)
mask[idx,] = False
con_sample,case_sample = x_train[idx], x_train[mask]


fprint("Separation succcessful")
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

fprint("After deletion")
fprint(x_train.shape)
fprint(x_test.shape)

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
    if fdr > 0.005:
        idx.append(i)
    i = i + 1

x_train = np.delete(x_train,idx,1)
x_test = np.delete(x_test,idx,1)


fprint("Filter succcessful")

# Checking
fprint(x_train.shape)
fprint(x_test.shape)
fprint(y_train.shape)
fprint(y_test.shape)


#Delete sparse sample

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

fprint("Deleted col sparse.")
fprint(x_train.shape)
fprint(x_test.shape)
fprint(y_train.shape)
fprint(y_test.shape)


# Save all files
df = pd.DataFrame(P_fdr)
df.to_csv(r"data/NEW_P_fdr.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(x_train)
df.to_csv(r"data/NEW_training_sample.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(x_test)
df.to_csv(r"data/NEW_testing_sample.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(y_train)
df.to_csv(r"data/NEW_training_label.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(y_test)
df.to_csv(r"data/NEW_testing_label.csv.gz",index=False,sep=",",compression="gzip")


# Checking

fprint("Finish!")


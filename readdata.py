import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 
from statsmodels.stats.multitest import fdrcorrection
from sklearn.model_selection import train_test_split
import gzip
import csv 

sample_transposed_f = r"data/out_tr.csv.gz" #transposed so each row is a sample
label_f = r"data/labely.csv.gz"


print("start") 

#read csv file, store in numpy
df = pd.read_csv(sample_transposed_f,compression ="gzip")
samples = df.to_numpy()

df = pd.read_csv(label_f,compression ="gzip")
label = df.to_numpy()

print("Samples ",samples.shape)
print("Labels ",label.shape)

#delete all zero columns and rows
idx = np.argwhere(np.all(samples[1:] ==0,axis = 0)) #find index of zero columns
samples = np.delete(samples,idx,axis = 1) 

print("After deletion of columns:",samples.shape)

idx = np.where(~samples.any(axis=1))[0]
samples = np.delete(samples,idx,axis = 0)
labels = np.delete(label,idx)

print("After deletion of rows:",samples.shape)

#Split RNA name with arest of array
rna_names = samples[0]
samples = samples[1:]

#split training and testing sample (x = sample, y = label )
x_train,x_test,y_train,y_test = train_test_split(samples,label,test_size = 0.1, random_state = 41)
print("Split succcessful")

#separate control and case samples
con_sample = np.empty((0,len(sample[0])))
case_sample = np.empty((0,len(sample[0])))
i = 0
for lb in label:
    if lb == "Control":
        con_sample = np.vstack([con_sample,sample[i]])
    else:
        case_sample = np.vstack([con_sample,sample[i]])
    i = i + 1

print("Separation succcessful")

# optain stat value
t_stat,pvalue = stats.ttest_ind(con_sample, case_sample, axis = 0, equal_var=True, nan_policy='raise')
rejected, P_fdr = fdrcorrection(pvalue, alpha=0.05, method='indep', is_sorted=False)

# select RNAs with small P_fdr as the input
i = 0
for fdr in P_fdr:
    if fdr > 0.02:
        x_train = np.delete(x_train,i,1)
        x_test = np.delete(x_test,i,1)
        rna_names = np.delete(rna_names,i,1)

print("Filter succcessful")

# Save all files
df = pd.DataFrame(x_train)
df.to_csv(r"data/training_sample.csv.gz",index=False,sep=" ",compression="gzip")
df = pd.DataFrame(x_test)
df.to_csv(r"data/testing_sample.csv.gz",index=False,sep=" ",compression="gzip")
df = pd.DataFrame(y_train)
df.to_csv(r"data/training_label.csv.gz",index=False,sep=" ",compression="gzip")
df = pd.DataFrame(y_test)
df.to_csv(r"data/testing_label.csv.gz",index=False,sep=" ",compression="gzip")
df = pd.DataFrame(rna_names)
df.to_csv(r"data/rna_name.csv.gz",index=False,sep=" ",compression="gzip")

# Checking
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(rna_names.shape)
print("Finish!")


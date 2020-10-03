import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
import gzip
import csv 

sample_transposed_f = r"data/out_tr.csv.gz" #each row is a sample
label_f = r"data/labely.csv"

# 1. cut column
# 2. perform T-test

print("start") 

#read csv file, store in numpy
df = pd.read_csv(sample_transposed_f,compression ="gzip")
samples = df.to_numpy()

df = pd.read_csv(label_f,compression ="gzip")
label = df.to_numpy()

print("Samples ",samples.shape)
print("Labels ",label.shape)

#delete all zero columns
idx = np.argwhere(np.all(samples[1:] ==0,axis = 0)) #find index of zero columns
samples = np.delete(samples,idx,axis = 1) 

print("After deletion:",samples.shape)

#Split RNA name with  rest of array
rna_names = samples[0]
samples = samples[1:]

#split training and testing sample (x = sample, y = label )
x_train,x_test,y_train,y_test = train_test_split(samples,label,test_size = 0.1, random_state = 41)

#separate control and case samples
con_sample = np.empty((0,len(sample[0])))
case_sample = np.empty((0,len(sample[0])))
for lb in label:
    if lb == "Control":
        con_sample = np.vstack([con_sample,sample[i]])
    else:

    




t_stat,pvalue = stats.ttest_ind(con_sample, case_sample, axis = 0, equal_var=True, nan_policy='raise')
rejected, P_fdr = sm.fdrcorrection(pvalue, alpha=0.05, method='indep', is_sorted=False)
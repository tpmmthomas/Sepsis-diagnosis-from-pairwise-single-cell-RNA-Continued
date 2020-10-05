import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 
from statsmodels.stats.multitest import fdrcorrection
from sklearn.model_selection import train_test_split
import gzip
import csv 

sample_transposed_f = r"/research/dept8/estr3108/cprj2716/out_tr.csv.gz" #transposed so each row is a sample
label_f = r"/research/dept8/estr3108/cprj2716/labely.csv.gz"

#sample_transposed_f = r"C:/Users/TPMMTHOMAS/Documents/GitHub/ESTR3108-Sepsis-diagnosis-from-pairwise-single-cell-RNA/data/out_tr.csv.gz" #transposed so each row is a sample
#label_f = r"C:/Users/TPMMTHOMAS/Documents/GitHub/ESTR3108-Sepsis-diagnosis-from-pairwise-single-cell-RNA/data/labely.csv.gz"
def fprint(txtt):
    f = open(r"/uac/cprj/cprj2716/dp.txt","a+")
    f.write(str(txtt))
    f.write("\n")
    f.close()

fprint("start") 

#read csv file, store in numpy

label = np.genfromtxt(label_f, delimiter=',', dtype=None, encoding=None,skip_header=0) 
fprint("input label")

samplesdf = pd.DataFrame()
for df in  pd.read_csv(sample_transposed_f,compression ="gzip", chunksize = 1000, header = 1):
    samplesdf = samplesdf.append(df)
samples = samplesdf.to_numpy()

fprint("Samples ")
fprint(samples.shape)
fprint("Labels ")
fprint(label.shape)

#Get RNA names
rna_names = pd.read_csv(sample_transposed_f,compression ="gzip", nrows=1 , header = 0)
rna_names = rna_names.to_numpy()
samples = samples.astype('float64')

fprint("Samples ")
fprint(samples.shape)
fprint("Labels ")
fprint(label.shape)


#delete all zero columns and rows
idx = np.argwhere(np.all(samples==0,axis = 0)) #find index of zero columns
samples = np.delete(samples,idx,axis = 1) 
rna_names = np.delete(rna_names,idx) 

fprint("After deletion of columns:")e
fprint(samples.shape)

idx = np.where(~samples.any(axis=1))[0]
samples = np.delete(samples,idx,axis = 0)
label = np.delete(label,idx)

fprint("After deletion of rows:")
fprint(samples.shape)

fprint("Labels check")
fprint(label.shape)

#split training and testing sample (x = sample, y = label)
x_train,x_test,y_train,y_test = train_test_split(samples,label,test_size = 0.1, random_state = 41)
fprint("Split succcessful")

#save 1 copy first
df = pd.DataFrame(x_train)
df.to_csv(r"/research/dept8/estr3108/cprj2716/training_sample_raw.csv.gz",index=False,sep=" ",compression="gzip")
df = pd.DataFrame(x_test)
df.to_csv(r"/research/dept8/estr3108/cprj2716/testing_sample_raw.csv.gz",index=False,sep=" ",compression="gzip")
df = pd.DataFrame(y_train)
df.to_csv(r"/research/dept8/estr3108/cprj2716/training_label_raw.csv.gz",index=False,sep=" ",compression="gzip")
df = pd.DataFrame(y_test)
df.to_csv(r"/research/dept8/estr3108/cprj2716/testing_label_raw.csv.gz",index=False,sep=" ",compression="gzip")
df = pd.DataFrame(rna_names)
df.to_csv(r"/reaksearch/dept8/estr3108/cprj2716/rna_names_raw.csv.gz",index=False,sep=" ",compression="gzip")

#checking 
fprint(x_train.shape)
fprint(x_test.shape)
fprint(y_train.shape)
fprint(y_test.shape)

#separate control and case samples
con_sample = np.empty((0,len(x_train[0])))
case_sample = np.empty((0,len(x_train[0])))
i = 0
for lb in y_train:
    if lb == "Control":
        con_sample = np.vstack([con_sample,x_train[i]])
    else:
        case_sample = np.vstack([case_sample,x_train[i]])
    i = i + 1

fprint("Separation succcessful")

df = pd.DataFrame(con_sample)
df.to_csv(r"/research/dept8/estr3108/cprj2716/con_sample.csv.gz",index=False,sep=" ",compression="gzip")
df = pd.DataFrame(case_sample)
df.to_csv(r"/research/dept8/estr3108/cprj2716/case_sample.csv.gz",index=False,sep=" ",compression="gzip")

# optain stat value
t_stat,pvalue = stats.ttest_ind(con_sample, case_sample, axis = 0, equal_var=True, nan_policy='raise')
rejected, P_fdr = fdrcorrection(pvalue, alpha=0.05, method='indep', is_sorted=False)

fprint("Value computed!")

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
df.to_csv(r"/research/dept8/estr3108/cprj2716/P_fdr.csv.gz",index=False,sep=" ",compression="gzip")
df = pd.DataFrame(x_train)
df.to_csv(r"/research/dept8/estr3108/cprj2716/training_sample.csv.gz",index=False,sep=" ",compression="gzip")
df = pd.DataFrame(x_test)
df.to_csv(r"/research/dept8/estr3108/cprj2716/testing_sample.csv.gz",index=False,sep=" ",compression="gzip")
df = pd.DataFrame(y_train)
df.to_csv(r"/research/dept8/estr3108/cprj2716/training_label.csv.gz",index=False,sep=" ",compression="gzip")
df = pd.DataFrame(y_test)
df.to_csv(r"/research/dept8/estr3108/cprj2716/testing_label.csv.gz",index=False,sep=" ",compression="gzip")
df = pd.DataFrame(rna_names)
df.to_csv(r"/research/dept8/estr3108/cprj2716/rna_name.csv.gz",index=False,sep=" ",compression="gzip")

# Checking
fprint(x_train.shape)
fprint(x_test.shape)
fprint(y_train.shape)
fprint(y_test.shape)
fprint(rna_names.shape)
fprint("Finish!")


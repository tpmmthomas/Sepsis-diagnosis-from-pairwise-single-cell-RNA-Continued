import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 
from statsmodels.stats.multitest import fdrcorrection
from sklearn.model_selection import train_test_split
import gzip
import csv 

fdr = r"data/P_fdr.csv.gz"
ytrain = r"data/training_label_raw.csv.gz"
xtest = r"data/testing_sample_raw.csv.gz"
ytest = r"data/testing_label_raw.csv.gz"


df =  pd.read_csv(fdr,compression ="gzip",delimiter=',',header =0)
print(df)

samples = df.to_numpy()

print(samples.shape)

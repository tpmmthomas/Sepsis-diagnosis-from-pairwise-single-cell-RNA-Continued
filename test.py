import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 
from statsmodels.stats.multitest import fdrcorrection
from sklearn.model_selection import train_test_split
import gzip
import csv 

xtrain = r"data/training_sample_raw.csv.gz"
ytrain = r"data/training_label_raw.csv.gz"
xtest = r"data/testing_sample_raw.csv.gz"
ytest = r"data/testing_label_raw.csv.gz"

samplesdf = pd.DataFrame()
for df in  pd.read_csv(xtest,compression ="gzip",delimiter=' ', chunksize = 1, header =0):
    samplesdf = samplesdf.append(df)
samples = samplesdf.to_numpy()


i = 0 
print(samples[0])
for sp in samples[0]:
    i = i + 1
print(i)
label = np.genfromtxt(ytest, delimiter=' ', dtype=None, encoding=None,skip_header=1) 

print(len(samples[0]))
print(samples.shape)
print(label[0])
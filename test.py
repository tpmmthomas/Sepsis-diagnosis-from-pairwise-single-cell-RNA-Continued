import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 
from statsmodels.stats.multitest import fdrcorrection
from sklearn.model_selection import train_test_split
import gzip
import csv 

def row_count(input):
    for df in  pd.read_csv(input,compression ="gzip", chunksize = 1, header =0):
        print(df)
        break

print("start")
xt_loc = r"data/training_sample_raw.csv.gz"
xtt_loc = r"data/testing_sample_raw.csv.gz"
yt_loc = r"data/training_label_raw.csv.gz"
ytt_loc = r"data/testing_label_raw.csv.gz"

row_count(xtt_loc)



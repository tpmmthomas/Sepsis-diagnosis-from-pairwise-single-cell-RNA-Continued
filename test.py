import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 
from statsmodels.stats.multitest import fdrcorrection
from sklearn.model_selection import train_test_split
import gzip
import csv 


xt_loc = r"data/training_sample_raw.csv.gz"
xtt_loc = r"data/testing_sample_raw.csv.gz"
yt_loc = r"data/training_label_raw.csv.gz"
ytt_loc = r"data/testing_label_raw.csv.gz"

print(row_count(ytt_loc))
print(row_count(yt_loc))
print(row_count(xtt_loc))
print(row_count(xt_loc))

def row_count(input):
    with gzip.open(input,"rt") as f:
        for i, l in enumerate(f):
            pass
    return i
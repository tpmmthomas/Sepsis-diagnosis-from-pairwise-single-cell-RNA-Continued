import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 
from statsmodels.stats.multitest import fdrcorrection
from sklearn.model_selection import train_test_split
import gzip
import csv 

label_f = r"C:/Users/TPMMTHOMAS/Documents/GitHub/ESTR3108-Sepsis-diagnosis-from-pairwise-single-cell-RNA/data/labely.csv.gz"
label = np.genfromtxt(label_f, delimiter=',', dtype=None, encoding=None,skip_header=0) 

for lb in label:
    print(lb)
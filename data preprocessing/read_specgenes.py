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
needed_genes = ["FCMR","PLAC8","PLA2G7","LAMP1","CEACAM4","NLRP1","IDNK"]

def fprint(txtt):
    f = open(r"dp1.txt","a+")
    f.write(str(txtt))
    f.write("\n")
    f.close()

fprint("start123") 

#read data
samplesdf =  pd.read_csv(sample_f,compression ="gzip", header = 0)
result_df = samplesdf[samplesdf['GENE'].isin(needed_genes)]
result_df = result_df.T
samples_name = result_df.index.values
samples_name = samples_name[1:]
samples = result_df.values
gene_name = samples[0]
df = pd.DataFrame(gene_name)
df.to_csv(r"data/name_index.csv",index=False,sep=",")
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


# Save all files
df = pd.DataFrame(samples)
df.to_csv(r"data/samplesSpec.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(labels)
df.to_csv(r"data/labelsSpec.csv.gz",index=False,sep=",",compression="gzip")

# Checking

fprint("Finish! Raw data")

FAIM3 = samplesdf.loc[samplesdf["GENE"] == "FCMR"]
FAIM3 = FAIM3.values
FAIM3 = FAIM3.flatten()[1:]
fprint("FAIM3:")
fprint(FAIM3.shape)

PLAC8 = samplesdf.loc[samplesdf["GENE"] == "PLAC8"]
PLAC8 = PLAC8.values
PLAC8 = PLAC8.flatten()[1:]
fprint("PLAC8:")
fprint(PLAC8.shape)


PLA2G7 = samplesdf.loc[samplesdf["GENE"] == "PLA2G7"]
PLA2G7 = PLA2G7.values
PLA2G7 = PLA2G7.flatten()[1:]
fprint("PLA2G7:")
fprint(PLA2G7.shape)

LAMP1 = samplesdf.loc[samplesdf["GENE"] == "LAMP1"]
LAMP1 = LAMP1.values
LAMP1 = LAMP1.flatten()[1:]
fprint("LAMP1:")
fprint(LAMP1.shape)

CEACAM4 = samplesdf.loc[samplesdf["GENE"] == "CEACAM4"]
CEACAM4 = CEACAM4.values
CEACAM4 = CEACAM4.flatten()[1:]
fprint("CEACAM4:")
fprint(CEACAM4.shape)

NLRP1 = samplesdf.loc[samplesdf["GENE"] == "NLRP1"]
NLRP1 = NLRP1.values
NLRP1 = NLRP1.flatten()[1:]
fprint("NLRP1:")
fprint(NLRP1.shape)

IDNK = samplesdf.loc[samplesdf["GENE"] == "IDNK"]
IDNK = IDNK.values
IDNK = IDNK.flatten()[1:]
fprint("IDNK:")
fprint(IDNK.shape)

outval = []
outy = []
i = 0
for x in FAIM3:
    y = PLAC8[i]
    if y == 0:
        i+=1
        continue
    outval.append(x/y)
    outy.append(labels[i])
    i += 1
fprint("FAIM3/PLAC8")
fprint(len(outval))
df = pd.DataFrame(outval)
df.to_csv(r"data/Comp_FAIM3dPLAC8.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(outy)
df.to_csv(r"data/Comp_FAIM3dPLAC8_y.csv.gz",index=False,sep=",",compression="gzip")

outval = []
outy = []
i = 0
for x in FAIM3:
    y = PLAC8[i]
    outval.append(x-y)
    outy.append(labels[i])
    i += 1
fprint("FAIM3-PLAC8")
fprint(len(outval))
df = pd.DataFrame(outval)
df.to_csv(r"data/Comp_FAIM3-PLAC8.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(outy)
df.to_csv(r"data/Comp_FAIM3-PLAC8_y.csv.gz",index=False,sep=",",compression="gzip")

outval = []
outy = []
i = 0
for x in PLAC8:
    y,z,a = PLA2G7[i],LAMP1[i],CEACAM4[i]
    if y==0 or a==0:
        i+=1
        continue
    outval.append((x/y)*(z/a))
    outy.append(labels[i])
    i += 1
fprint("SeptiCyte1")
fprint(len(outval))
df = pd.DataFrame(outval)
df.to_csv(r"data/Comp_SeptiCyte1.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(outy)
df.to_csv(r"data/Comp_SeptiCyte1_y.csv.gz",index=False,sep=",",compression="gzip")

outval = []
outy = []
i = 0
for x in PLAC8:
    y,z,a = PLA2G7[i],LAMP1[i],CEACAM4[i]
    outval.append(x-y+z-a)
    outy.append(labels[i])
    i += 1
fprint("SeptiCyte2")
fprint(len(outval))
df = pd.DataFrame(outval)
df.to_csv(r"data/Comp_SeptiCyte2.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(outy)
df.to_csv(r"data/Comp_SeptiCyte2_y.csv.gz",index=False,sep=",",compression="gzip")

outval = []
outy = []
i = 0
for x in NLRP1:
    y,z = IDNK[i],PLAC8[i]
    if z == 0:
        i += 1
        continue
    outval.append((x-y)/z)
    outy.append(labels[i])
    i += 1
fprint("SNIP1")
fprint(len(outval))
df = pd.DataFrame(outval)
df.to_csv(r"data/Comp_SNIP1.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(outy)
df.to_csv(r"data/Comp_SNIP1_y.csv.gz",index=False,sep=",",compression="gzip")

outval = []
outy = []
i = 0
for x in NLRP1:
    y,z = IDNK[i],PLAC8[i]
    outval.append((x-y)-z)
    outy.append(labels[i])
    i += 1
fprint("SNIP2")
fprint(len(outval))
df = pd.DataFrame(outval)
df.to_csv(r"data/Comp_SNIP2.csv.gz",index=False,sep=",",compression="gzip")
df = pd.DataFrame(outy)
df.to_csv(r"data/Comp_SNIP2_y.csv.gz",index=False,sep=",",compression="gzip")

fprint("Finish!")
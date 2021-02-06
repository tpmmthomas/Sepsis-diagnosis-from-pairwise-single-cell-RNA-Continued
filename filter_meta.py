import numpy as np
import pandas as pd

samples = pd.read_csv("research/data/scp_gex_matrix.csv.gz")
labels 	= pd.read_csv("research/data/scp_meta.txt")
print("shape of samples",samples.shape)
print("shape of labels", labels.shape)

samples_T = samples.T
print("shape of transposed samples",samples_T.shape)

samples_gene = samples_T.index #samples_gene[0] is containing 'GENE'
print("shape of samples_gene",samples_gene) #length: 122558

meta_gene = labels['NAME'] #meta_gene[0] is containing 'TYPE'
print("shape of meta_gene",meta_gene) #length: 126352

#check if duplication exist
if samples_gene.duplicated().sum() >0:
	exit()
else:
	print("samples_gene contains no duplicate")

if meta_gene.duplicated().sum() >0:
	exit()
else:
	print("meta_gene contains no duplciate")

#filtering
gene_filter = meta_gene.isin(samples_gene)
print("number of gene shared in both file:",gene_filter.sum())

#apply result to meta.txt and generate new file
gene_filter[0]=True #gene_filter[0] is header row e.g. 'GENE' and 'TYPE'
filter_meta = labels[gene_filter]
print("shape of filitered meta.txt")

#output new meta.txt
filter_meta.to_csv(r"research/data/filtered_meta.txt",index=None,sep="\t",mode='w')
print("SUCCEED")


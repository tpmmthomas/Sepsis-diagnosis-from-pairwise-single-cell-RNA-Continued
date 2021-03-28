# Sepsis-diagnosis-from-pairwise-single-cell-RNA

Store all related codes regarding this project.


## Data PreProcessing 
-- Please check "data preprocessing" folder.

`readdata.py` contains related codes for data preprocessing stage.   
`changefdrlimit.py` changes the Fdr Threshold during experiment.   
`deleteSparseData.py` changes the Sparse Threshold during experiment.   
`read_specgenes.py` selects specific genes for comparison with other biomarkers.
 
 ## Model Training
`3s_series_trTran_CNN` contains related codes for our chosen model in the course. (Transformer + CNN, ESTR3108)
`NEW_NEW_capsnet.ipynb` contains related codes for our newest model. (CapsuleNet + Transformer)

 ## Model Comparison
the folder "OTHER_MODELS" shows other models we have trained for comparison. (May include same model but with different parameters)

## Results
Screen captures and graphs for evaluating the model is put in the "Results" folder.




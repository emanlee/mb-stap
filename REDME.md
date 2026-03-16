# MB-STAP

MB-STAP is a lightweight supervised deep learning model designed to predict causal regulatory relationships between genes from time-series single-cell RNA sequencing data. 

Code is programmed using Python 3.8.

Full code and data are available at https://sourceforge.net/projects/mb-stap/ .


## Data

This study evaluated MB-STAP on the following four datasets:

mESC1: Mouse embryonic stem cells at 4 time points, comprising 2,718 cells (accession number: GSE65525).
mESC2: Mouse embryonic stem cells at 9 time points, comprising 3,456 cells (accession number: GSE79578).
hESC1: Human embryonic stem cells at 5 time points, comprising 1,530 cells (accession number: E-MTAB-3929).
hESC2: Human embryonic stem cells at 6 time points, comprising 758 cells (accession number: GSE75748).

The benchmark data and the preprocessed gene expression matrices for these datasets can be downloaded from this project repository.
We downloaded the single-cell mouse pancreatic dataset with accession number GSE87375, identified the top-ranked predicted klf10 and klf11, constructed gene pairs associated with them, and performed gene causal relationship analysis using MB-STAP.

## TASK 1, Evaluating MB-STAP on four datasets


### STEP 1: Constructing three-dimensional histograms

**Code**: set_3Dhistogram.py

**Input**: Gene expression profile, Gene pair files, gene index files, etc.

**Command for each cell type**:
```
python /home/yanke/set_3Dhistogram.py     None     /home/yanke/data/hesc1_gene_list_ref.txt     /home/yanke/data/hesc1_gene_pairs_400.txt     /home/yanke/data/hesc1_gene_pairs_400_num.txt     None     "/home/yanke/scRNA_expression_data/hesc1_expression_data/RPKM"     5     1     /home/yanke/data/GTRD_NT_8X8_5_hesc1
python /home/yanke/set_3Dhistogram.py     None     /home/yanke/data/hesc2_gene_list_ref.txt     /home/yanke/data/hesc2_gene_pairs_400.txt     /home/yanke/data/hesc2_gene_pairs_400_num.txt     None     "/home/yanke/scRNA_expression_data/hesc2_expression_data/RPKM"     6     1     /home/yanke/data/ GTRD_NT_8X8_6_hesc2
python /home/yanke/set_3Dhistogram.py     None     /home/yanke/data/mesc1_gene_list_ref.txt     /home/yanke/data/mesc1_gene_pairs_400.txt     /home/yanke/data/mesc1_gene_pairs_400_num.txt     None     "/home/yanke/scRNA_expression_data/mesc1_expression_data/RPKM"     4     1     /home/yanke/data/ GTRD_NT_8X8_9_mesc1
python /home/yanke/set_3Dhistogram.py     None     /home/yanke/data/mesc2_gene_list_ref.txt     /home/yanke/data/mesc2_gene_pairs_400.txt     /home/yanke/data/mesc2_gene_pairs_400_num.txt     None     "/home/yanke/scRNA_expression_data/mesc2_expression_data/RPKM"     9     1     /home/yanke/data/ GTRD_NT_8X8_4_mesc2

```

**output**:

- x file: The representation of genes' expression file, use as the input of the model.
- y file: The label for the corresponding pairs.
- z file: Indicate the gene name for each pair.


### STEP 2: Three-fold Cross-validation for MB-STAP

**Code**: MB-STAP.py

**Input**: the output of the STEP 1.

**usage**:

```
python MB-STAP.py
```


## TASK 2, Inferring the Gene Regulatory Networks of klf10 and klf11 Using MB-STAP


### STEP 1: Convert the downloaded dataset into h5 file. 

**Code**: makeH5.py

### STEP 2: Predict use trained MB-STAP

**Code**: predict.py

**usage**:

```
python predict.py 2 /home/yanke/MB/predict/Nx 3 /home/yanke/MB/predict/keras_cnn_trained_model_shallow.h5

```
### STEP 3: Process post-prediction files

**Code**: tf_predictions_by_segment.py

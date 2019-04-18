# Network Attack Outlierness
Study of the outlierness properties of network traffic attacks.

Experiments conducted for the paper:  
"Are Network Attacks Outliers? A Study of Space Representations and Unsupervised Algorithms"

April, 2019  
Félix Iglesias, TU Wien  
Alexander Hartl, TU Wien


## Summary 
The following experiments use the CICIDS2017 dataset for evaluating the suitability of different feature vectors (Consensus, CAIA, AGM, TA and Cisco) and outlier detection methods (HBOS, LOF, kNN, iForest and SDO) for detecting network attacks. With this research we aimed to answer the question if network attacks constitute outliers in the sense of unsupervised anomaly detection, and to find the most suitable feature representations for attack detection.
The experiments consist of three steps:
* Hyperparameter search to find the optimal parameters for each feature vector-outlier detection method combination
* Benchmarking of the outlier detection methods with respect to several established performance indices
* A forward feature selection to find the feature vector that optimizes detection performance using a joint CAIA-Consensus-AGM feature vector as basis

## Requirements 
1. The CICIDS2017 dataset belongs to the Canadian Institute for Cybersecurity. It can be obtained on demand
in the following link:
https://www.unb.ca/cic/datasets/ids-2017.html

2. CICIDS2017 is provided as pcaps and a ground truth file. We extracted several feature vectors with a feature extractor 
based on Golang. Our feature extractor is pending publication (shortly), but it can be requested on demand by
email here: **felix.iglesias(at)nt.tuwien.ac.at**. In any case, features can be extracted with any feature extractor, 
e.g., tshark. 




## Instructions for replication 
1. Fetch a copy of the repository:  
`git clone https://github.com/CN-TU/network-attack-outlierness.git && cd network-attack-outlierness`

2. Make sure you have all required python packages:  
`pip3 install -r requirements.txt`

3. Copy to the preprocessed data (labeled feature vectors as csv files) to the working directory. 

4. Run the experiments:  
`make`

Results are placed in the following folders:

Folder | Description
-------|-------------
hist | Histograms visualizing outlier scores of normal and attack traffic
histlog | Same as hist, but in logarithmic scale
results | Performance indices for all outlier detection methods as csv files
scores | Outlier scores as NumPy arrays for subsequent analysis
stats | Statistics of outlier scores of normal and attack traffic as csv files
tuneres | Outputs of script for parameter search

Please note that some of the feature vectors and/or outlier detection algorithms require substantial amounts of memory
and/or CPU time. To run the experiments for just one feature vector you can replace step 3 by (e.g. for OptOut)  
`make OptOut `


## Included files 

File | Description
-----|------------- 
README.txt | This file
Makefile | Makefile for reproducing all results
od.py | Script for evaluating outlier detection methods
tune.py | Script for hyperparameter selection for different methods
find_vector.py | Script for performing the forward feature selection
requirements.txt | List of required python packages
[outdet] | Python package providing unified wrapper functions for outlier detection algorithms

--------

More information about our research at the CN-Group TU Wien:

https://www.cn.tuwien.ac.at/

# Network Attack Outlierness
Study of the outlierness properties of network traffic attacks.

Experiments conducted for the paper:  
"Are Network Attacks Outliers? A Study of Space Representations and Unsupervised Algorithms"

April, 2019  
Félix Iglesias, TU Wien  
Alexander Hartl, TU Wien


## Summary 
The following experiments use the CICIDS2017 dataset for evaluating the suitability of different feature vectors and
outlier detection methods for detecting network attacks.

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

3. Run the experiments:  
`make`

Results are placed in the following folders:

Folder | Description
-------|-------------
hist | Histograms visualizing outlier scores of normal and attack traffic
histlog | Histograms in logarithmic scale
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
od.py | Script for performing 
tune.py | Script for parameter selection for different methods
find_vector.py | Script for performing a forward feature selection
[outdet] | Python package providing unified wrapper function for outlier detection algorithms

--------

More information about our research at the CN-Group TU Wien:

https://www.cn.tuwien.ac.at/

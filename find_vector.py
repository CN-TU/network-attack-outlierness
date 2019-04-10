#!/usr/bin/env python3

import outdet
import numpy as np
import pandas as pd

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from multiprocessing import Pool

import collections

REPEATS = 10

np.random.seed(2)

data = pd.read_csv("CAIAConAGM.csv").fillna(0)
#data = pd.read_csv('AGM.csv').fillna(0)
attacks = data['Attack']
labels = np.array(data['Label'])

# CAIAConAGM
data = data.drop (columns=[
	'flowStartMilliseconds',
	'sourceIPAddress',
	'destinationIPAddress',
	'mode(destinationIPAddress)',
	'mode(_tcpFlags)',
	'Label',
	'Attack' ])
	
nominalFeatures = [
	'sourceTransportPort',
	'destinationTransportPort',
	'mode(sourceTransportPort)',
	'mode(destinationTransportPort)',
	'protocolIdentifier' ]

# AGM
#data = data.drop (columns=[
	#'flowStartMilliseconds',
	#'sourceIPAddress',
	#'mode(destinationIPAddress)',
	#'mode(_tcpFlags)',
	#'Label',
	#'Attack' ])
	
#nominalFeatures = ['mode(sourceTransportPort)', 'mode(destinationTransportPort)', 'mode(protocolIdentifier)']

for nominal in nominalFeatures:
	freqValues = list(data[nominal].value_counts().iloc[:10].keys())
	data.loc[~data[nominal].isin(freqValues),nominal] = np.nan
data = pd.get_dummies (data, columns = nominalFeatures, drop_first = True, dtype=np.float64)

columns = list(data.columns)
allNominal = []
nominalColumns = {}
for feat in nominalFeatures:
	nominalColumns[feat] = [ i for i in range(len(columns)) if columns[i].startswith(feat+'_') ]
	allNominal +=  nominalColumns[feat]
	
vec = [False] * len(columns)
notAdded = [ ind for ind in range(len(columns)) if ind not in allNominal ]
notAddedNominal = nominalFeatures[:]

# downsample to 5%
_, data, _, labels = train_test_split (data, labels, test_size=0.05, random_state = 1, stratify=attacks)
data = minmax_scale (data)

	
overallBestScore = 0
overallBest = None

while not all(vec):
	bestScoreThisRound = 0
	bestThisRound = None
	
	def dof(feature):
		locVec = vec[:]
		
		print ('.', end='', flush = True)
		if isinstance(feature, str):
			for ind in nominalColumns[feature]:
				locVec[ind] = True
		else:
			locVec[feature] = True

		return sum((roc_auc_score(labels, outdet.SDO(data[:,locVec], k=200,  return_scores=True)) for _ in range(REPEATS))) / REPEATS
	
	with Pool(16) as p:
		results = p.map(dof, notAddedNominal + notAdded)
	
	ind = max(range(len(notAddedNominal) + len(notAdded)), key=lambda x: results[x])
	bestScoreThisRound = results[ind]
	if ind < len(notAddedNominal):
		bestThisRound = notAddedNominal[ind]
		notAddedNominal = [ feat for feat in notAddedNominal if feat != bestThisRound ]
		for i in nominalColumns[bestThisRound]:
			vec[i] = True
	else:
		bestThisRound = columns[notAdded[ind - len(notAddedNominal)]]
		vec[notAdded[ind - len(notAddedNominal)]] = True
		notAdded = [ feat for feat in notAdded if feat != notAdded[ind - len(notAddedNominal)] ]
		
	print ("\nAdding feature %s with ROC-AUC %.4f" % (bestThisRound, bestScoreThisRound))
		
	

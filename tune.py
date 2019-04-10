#!/usr/bin/env python3

import numpy as np
import pandas as pd

import outdet

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split, ShuffleSplit

from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.metrics import roc_auc_score

import sys

np.random.seed(1)

data = pd.read_csv('%s.csv' % ('CAIAConAGM' if sys.argv[1] == 'OptOut' else sys.argv[1])).fillna(0)
attacks = data['Attack']
labels = np.array(data['Label'])

downsample = 0.05

if sys.argv[1] =='Consensus':
	data = data.drop (columns=[
		'flowStartMilliseconds',
		'sourceIPAddress',
		'destinationIPAddress',
		'Label',
		'Attack' ])
	nominalFeatures = ['sourceTransportPort', 'destinationTransportPort', 'protocolIdentifier']
	for nominal in nominalFeatures:
		freqValues = list(data[nominal].value_counts().iloc[:10].keys())
		data.loc[~data[nominal].isin(freqValues),nominal] = np.nan
	data = pd.get_dummies (data, columns = nominalFeatures, drop_first = True, dtype=np.float64)
	
elif sys.argv[1] == 'CAIA':
	data = data.drop (columns=[
		'flowStartMilliseconds',
		'sourceIPAddress',
		'destinationIPAddress',
		'sourceTransportPort',
		'destinationTransportPort',
		'Label',
		'Attack' ])
	data = pd.get_dummies (data, columns = ['protocolIdentifier'], drop_first = True, dtype=np.float64)
	
elif sys.argv[1] == 'AGM':
	data = data.drop (columns=[
		'flowStartMilliseconds',
		'sourceIPAddress',
		'mode(destinationIPAddress)',
		'mode(_tcpFlags)',
		'Label',
		'Attack' ])
	nominalFeatures = ['mode(sourceTransportPort)', 'mode(destinationTransportPort)', 'mode(protocolIdentifier)']
	for nominal in nominalFeatures:
		freqValues = list(data[nominal].value_counts().iloc[:10].keys())
		data.loc[~data[nominal].isin(freqValues),nominal] = np.nan
	data = pd.get_dummies (data, columns = nominalFeatures, drop_first = True, dtype=np.float64)
	
elif sys.argv[1] == 'TA':
	data = data.drop (columns=[
		'flowStartMilliseconds',
		'sourceIPAddress',
		'destinationIPAddress',
		'sourceTransportPort',
		'destinationTransportPort',
		'__NTAFlowID',
		'Label',
		'Attack' ])
	nominalFeatures = ['__NTAPorts', '__NTAProtocol']
	for nominal in nominalFeatures:
		freqValues = list(data[nominal].value_counts().iloc[:10].keys())
		data.loc[~data[nominal].isin(freqValues),nominal] = np.nan
	data = pd.get_dummies (data, columns = nominalFeatures, drop_first = True, dtype=np.float64)
	
elif sys.argv[1] == 'CAIAConAGM':
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
	for nominal in nominalFeatures:
		freqValues = list(data[nominal].value_counts().iloc[:10].keys())
		data.loc[~data[nominal].isin(freqValues),nominal] = np.nan
	data = pd.get_dummies (data, columns = nominalFeatures, drop_first = True, dtype=np.float64)
	
elif sys.argv[1] == 'Cisco':
	downsample = 0.01
	data = data.drop (columns=[
		'time_start',
		'sa',
		'da',
		'Label',
		'Attack' ])
	nominalFeatures = ['sp', 'dp', 'pr']
	for nominal in nominalFeatures:
		freqValues = list(data[nominal].value_counts().iloc[:10].keys())
		data.loc[~data[nominal].isin(freqValues),nominal] = np.nan
	data = pd.get_dummies (data, columns = nominalFeatures, drop_first = True, dtype=np.float64)
	
		
elif sys.argv[1] == 'OptOut':
	data = pd.get_dummies (data, columns = ['protocolIdentifier'], drop_first = True, dtype=np.float64)
	data = data[[
		'modeCount(destinationIPAddress)',
		'distinct(ipTTL)',
		'apply(packetTotalCount,forward)',
		'apply(stdev(ipTotalLength),backward)',
		'apply(octetTotalCount,forward)']]

else:
	print ('Unknown feature vector')
	exit
		
# downsample to 5%
_, data, _, labels = train_test_split (data, labels, test_size=downsample, random_state = 2, stratify=attacks)
data = minmax_scale (data)


def find_parameter(func, minN, maxN):
	cache = {}
	while True:
		todo = np.unique(np.round(np.geomspace(minN, maxN, 10)))
		results = []
		for n in todo:
			if n in cache:
				score = cache[n]
			else:
				score = roc_auc_score (labels, func(int(n)))
				cache[n] = score
			print ("ROC-AUC for %d: %.4f" % (n, score))
			results.append(score)
			
		best = max(range(len(results)), key=lambda i: results[i])
		newMinN = todo[best-1] if best > 0 else todo[best]
		newMaxN = todo[best+1] if best+1 < len(results) else todo[best]
		if newMinN == minN and newMaxN == maxN:
			break
		minN, maxN = newMinN, newMaxN

	best = max(cache, key=lambda x: cache[x])
	print ('Best: %d with ROC-AUC %.4f' % (best, cache[best]))

print ('\nKNN\n----------------')
find_parameter (lambda x: outdet.KNN(data, n_neighbors = x, return_scores=True), 2, 15)

print ('\nLOF\n----------------')
find_parameter (lambda x: outdet.LOF(data, n_neighbors = x, contamination=0.1, return_scores=True), 5, 50)

print ('\nHBOS\n----------------')
find_parameter (lambda x: outdet.HBOS(data, n_bins = x, return_scores = True), 20, 15000)

print ('\nIsolationForest\n----------------')

class odIfEstimator:
	def __init__(self, **kwargs):
		self.set_params (**kwargs)
	def get_params(self, deep = True):
		return self.params
	def set_params(self, **kwargs):
		self.params = kwargs
	def fit(self, X, y):
		pass
	def score(self, Xt, yt):
		return roc_auc_score(yt, outdet.IsolationForest(Xt, contamination=0.1, behaviour='new', return_scores = True, **self.params))
	
params = {
	'max_samples': list(range(100, 1000)),
	'n_estimators': list(range(50,200)),
	'max_features': list(range(1, data.shape[1])) }

cv = EvolutionaryAlgorithmSearchCV (
	estimator = odIfEstimator(),
	params = params,
	gene_type = [2, 2, 2],
	verbose = 1,
	population_size = 80,
	gene_mutation_prob = .1,
	gene_crossover_prob = .5,
	tournament_size = 3,
	generations_number = 8,
	# this is already validation set, no need for cross validation
	cv = ShuffleSplit(test_size=0.99, n_splits=1),
	n_jobs = 40)
	
cv.fit(data, labels)


params = {
	'k': list(range(100,1000)),
	'x': list(range(3,30)),
	'qv': [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] }
	
print ('\nSDO\n----------------')
class odSDOEstimator:
	def __init__(self, **kwargs):
		self.set_params (**kwargs)
	def get_params(self, deep = True):
		return self.params
	def set_params(self, **kwargs):
		self.params = kwargs
	def fit(self, X, y):
		pass
	def score(self, Xt, yt):
		return sum(roc_auc_score(yt, outdet.SDO(Xt, chunksize=10000, return_scores = True, **self.params)) for _ in range(5)) / 5
	

cv = EvolutionaryAlgorithmSearchCV (
	estimator = odSDOEstimator(),
	params = params,
	gene_type = [2, 2, 2],
	verbose = 1,
	population_size = 80,
	gene_mutation_prob = .1,
	gene_crossover_prob = .5,
	tournament_size = 3,
	generations_number = 8,
	# this is already validation set, no need for cross validation
	cv = ShuffleSplit(test_size=0.99, n_splits=1),
	n_jobs = 40)
	
cv.fit(data, labels)


print ('\nSDO w/ hbs\n----------------')


params = {
	'k': list(range(100,1000)),
	'x': list(range(3,30)),
	'qv': [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
	'hbs': [True] }
	
cv = EvolutionaryAlgorithmSearchCV (
	estimator = odSDOEstimator(),
	params = params,
	gene_type = [2, 2, 2],
	verbose = 1,
	population_size = 80,
	gene_mutation_prob = .1,
	gene_crossover_prob = .5,
	tournament_size = 3,
	generations_number = 8,
	# this is already validation set, no need for cross validation
	cv = ShuffleSplit(test_size=0.99, n_splits=1),
	n_jobs = 40)
	
cv.fit(data, labels)

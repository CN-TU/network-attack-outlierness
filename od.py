#!/usr/bin/env python3

import numpy as np
import pandas as pd

import outdet
import outdet.indices

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

import matplotlib
import matplotlib.pyplot as plt

import collections
import sys
import time
import resource
import json

np.random.seed(1)

REPEATS = 10

mode = 'all'
if len(sys.argv) > 2 and sys.argv[2] in ['all', 'pre', 'nopre']:
	mode = sys.argv[2]
	
inputfile = '%s' % ('CAIAConAGM' if sys.argv[1] == 'OptOut' else sys.argv[1])
algorithm = sys.argv[3] if len(sys.argv) > 3 else None

if mode == 'nopre':
	data = np.load('%s_data.npy' % sys.argv[1])
	labels = np.load('%s_labels.npy' % sys.argv[1])
	
else:
	data = pd.read_csv('%s.csv' % inputfile).fillna(0)
	attacks = data['Attack']
	labels = np.array(data['Label'])

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
	#_, data, _, labels = train_test_split (data, labels, test_size=0.05, random_state = 1, stratify=attacks)
	data = minmax_scale (data)
	
	if mode == 'pre':
		np.save('%s_data.npy' % sys.argv[1], data)
		np.save('%s_labels.npy' % sys.argv[1], labels)
		exit (0)
		
if sys.argv[1] =='Consensus':
	KNNparams = {'n_neighbors': 2}
	LOFparams = {'n_neighbors': 5}
	HBOSparams = {'n_bins': 20}
	IFparams = {'max_samples': 860, 'n_estimators': 50, 'max_features': 37}
	SDOparams = {'k': 553, 'x': 9, 'qv': 0.2, 'hbs': True}
	
elif sys.argv[1] == 'CAIA':
	KNNparams = {'n_neighbors': 2}
	LOFparams = {'n_neighbors': 5}
	HBOSparams = {'n_bins': 22}
	IFparams = {'max_samples': 873, 'n_estimators': 95, 'max_features': 26}
	SDOparams = {'k': 396, 'x': 5, 'qv': 0.25, 'hbs': True}
	
elif sys.argv[1] == 'AGM':
	KNNparams = {'n_neighbors': 15}
	LOFparams = {'n_neighbors': 18}
	HBOSparams = {'n_bins': 992}
	IFparams = {'max_samples': 696, 'n_estimators': 96, 'max_features': 2}
	SDOparams = {'k': 823, 'x': 11, 'qv': 0.2}
	
elif sys.argv[1] == 'TA':
	KNNparams = {'n_neighbors': 3}
	LOFparams = {'n_neighbors': 5}
	HBOSparams = {'n_bins': 21}
	IFparams = {'max_samples': 529, 'n_estimators': 64, 'max_features': 1}
	SDOparams = {'k': 926, 'x': 23, 'qv': 0.5, 'hbs': True}
	
elif sys.argv[1] == 'CAIAConAGM':
	KNNparams = {'n_neighbors': 15}
	LOFparams = {'n_neighbors': 50}
	HBOSparams = {'n_bins': 20}
	IFparams = {'max_samples': 169, 'n_estimators': 61, 'max_features': 72}
	SDOparams = {'k': 148, 'x': 28, 'qv': 0.5}
	
elif sys.argv[1] == 'Cisco':
	KNNparams = {'n_neighbors': 15}
	LOFparams = {'n_neighbors': 39}
	HBOSparams = {'n_bins': 20}
	IFparams = {'max_samples': 855, 'n_estimators': 73, 'max_features': 428}
	SDOparams = {'k': 281, 'x': 11, 'qv': 0.2, 'hbs': True}
	
	
elif sys.argv[1] == 'OptOut':
	KNNparams = {'n_neighbors': 15}
	LOFparams = {'n_neighbors': 50}
	HBOSparams = {'n_bins': 22}
	IFparams = {'max_samples': 456, 'n_estimators': 50, 'max_features': 4}
	SDOparams = {'k': 241, 'x': 25, 'qv': 0.35}
	
else:
	print ('Unknown feature vector')
	exit


matplotlib.rcParams.update({'font.size': 20})

results = collections.defaultdict(collections.Counter)
stds = collections.defaultdict(collections.Counter)
stats = collections.defaultdict(dict)

def process_scores(name, scores):
	global results, boxplot_counter, boxplot_labels, stds, stats
	
	indices = outdet.indices.get_indices(scores, labels)
	indices['spearmanr'] = spearmanr (scores, labels).correlation
	
	if name not in results:
		minScore = np.min(scores)
		maxScore = np.max(scores)
		
		np.save('scores/%s_%s.npy' % (sys.argv[1], name), scores)
	
		stats[name]['quartile10_attack'] = np.quantile(scores[labels==1], 0.1)
		stats[name]['quartile10_noattack'] = np.quantile(scores[labels==0], 0.1)
		stats[name]['quartile90_attack'] = np.quantile(scores[labels==1], 0.9)
		stats[name]['quartile90_noattack'] = np.quantile(scores[labels==0], 0.9)
		stats[name]['mean_attack'] = np.mean(scores[labels==1])
		stats[name]['mean_noattack'] = np.mean(scores[labels==0])
		stats[name]['median_attack'] = np.median(scores[labels==1])
		stats[name]['median_noattack'] = np.median(scores[labels==0])
		stats[name]['std_attack'] = np.std(scores[labels==1])
		stats[name]['std_noattack'] = np.std(scores[labels==0])
		
		maxPlot = np.quantile(scores, 0.95)

		plt.clf()
		plt.hist(scores[labels==0], density=True, range=(minScore, maxPlot), alpha=0.8, bins=50, color='#1f77b4', label='Normal traffic')
		plt.hist(scores[labels==1], density=True, range=(minScore, maxPlot), alpha=0.8, bins=50, color='#ff7f0e', label='Attacks')
		plt.legend(loc='upper right')
		plt.ylabel('Empirical pdf')
		plt.xlabel('Outlier score')
		plt.title('%s-%s' % (sys.argv[1], name))
		plt.tight_layout()
		plt.savefig('hist/%s_%s.png' % (sys.argv[1], name))
		plt.savefig('hist/%s_%s.pdf' % (sys.argv[1], name))
		plt.clf()
		plt.hist(scores[labels==0], density=True, range=(minScore, maxPlot), alpha=0.8, log=True, bins=50, color='#1f77b4', label='Normal traffic')
		plt.hist(scores[labels==1], density=True, range=(minScore, maxPlot), alpha=0.8, log=True, bins=50, color='#ff7f0e',  label='Attacks')
		plt.legend(loc='upper right')
		plt.xlabel('Outlier score')
		plt.ylabel('Empirical pdf')
		plt.title('%s-%s' % (sys.argv[1], name))
		plt.tight_layout()
		plt.savefig('histlog/%s_%s.png' % (sys.argv[1], name))
		plt.savefig('histlog/%s_%s.pdf' % (sys.argv[1], name))
	results[name].update (indices)
	stds[name].update({x+'_std': y**2 for x,y in indices.items()})

elapsed = 0

for k in range(REPEATS):
	print ('.', end='', flush=True)
	if algorithm == 'SDO' or not algorithm:
		start = time.time()
		scores = outdet.SDO(data, **SDOparams, return_scores = True)
		elapsed += time.time() - start
		process_scores ('SDO', scores)
		
	if algorithm == 'IF' or not algorithm:
		start = time.time()
		scores = outdet.IsolationForest(data, **IFparams, contamination=0.1, behaviour='new', return_scores=True)
		elapsed += time.time() - start
		process_scores ('IF', scores)

if algorithm == 'KNN' or not algorithm:
	start = time.time()
	scores = outdet.KNN(data, **KNNparams, algorithm='kd_tree', leaf_size=30, contamination=0.1, return_scores = True)
	elapsed += time.time() - start
	process_scores ('KNN', scores)
	
if algorithm == 'LOF' or not algorithm:
	start = time.time()
	scores = outdet.LOF(data, **LOFparams, algorithm='kd_tree', leaf_size=30, contamination=0.1, return_scores = True)
	elapsed += time.time() - start
	process_scores ('LOF', scores)
	
if algorithm == 'HBOS' or not algorithm:
	start = time.time()
	scores = outdet.HBOS(data, **HBOSparams, contamination=0.1, return_scores = True)
	elapsed += time.time() - start
	process_scores ('HBOS', scores)
	
	
print ('Execution time: %.3f' % elapsed)
print ('Memory consumption: %d' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

# reorder and write
df = pd.concat((pd.DataFrame(results), pd.DataFrame(stds)), axis=0).T
for alg in 'KNN', 'LOF', 'HBOS':
	if alg in df.index:
		df.loc[alg] = df.loc[alg] * REPEATS
	
for column in df.columns:
	if not column.endswith('_std'):
		df[column] = df[column] / REPEATS
		df[column + '_std'] = ((df[column + '_std'] - REPEATS * df[column]**2) / (REPEATS-1)).pow(0.5)

filename = ('results/%s_%s_results.csv' % (sys.argv[1], algorithm)) if algorithm else ('%s_results.csv' % sys.argv[1])
columns = ['spearmanr', 'Patn', 'adj_Patn', 'ap', 'adj_ap', 'maxf1', 'adj_maxf1', 'auc']
df[[x+tag for x in columns for tag in ['', '_std']]].to_csv(filename)
		
df = pd.DataFrame(stats).T
filename = ('stats/%s_%s_stats.csv' % (sys.argv[1], algorithm)) if algorithm else ('%s_stats.csv' % sys.argv[1])
columns = ['quartile10_attack', 'quartile10_noattack', 'quartile90_attack', 'quartile90_noattack', 'mean_attack', 'mean_noattack', 'median_attack', 'median_noattack', 'std_attack', 'std_noattack']
df[columns].to_csv(filename)

"""
A package providing unified wrapper functions for various outlier detection frameworks.
"""

import sklearn.neighbors
import sklearn.ensemble
import sklearn.cluster

import pyod.models.knn
import pyod.models.abod
import pyod.models.hbos
import pyod.models.loci

import PyNomaly.loop
import hdbscan

import math
import numpy as np
import scipy.spatial.distance as distance

def AutoEnc(X, return_scores = False, **kwargs):
	import pyod.models.auto_encoder
	ae = pyod.models.auto_encoder.AutoEncoder(**kwargs)
	is_outlier = ae.fit_predict(X)
	if return_scores:
		return ae.decision_scores_
	return is_outlier == 1

def LOOP(X, return_scores = False, contamination=0.1, **kwargs):
	[m, n] = X.shape
	loop = PyNomaly.loop.LocalOutlierProbability(X)
	scores = loop.fit().local_outlier_probabilities
	if return_scores:
		return scores
	scores_sorted = np.sort(scores)
	threshold = scores_sorted[-math.floor(m*contamination)-1]
	return scores > threshold

def LOF(X, return_scores = False, **kwargs):
	lof = sklearn.neighbors.LocalOutlierFactor(**kwargs)
	lof.fit (X)
	if return_scores:
		return -lof.negative_outlier_factor_
	is_inlier = lof.fit_predict (X)
	return is_inlier == -1
	
def IsolationForest(X, return_scores = False, **kwargs):
	IF = sklearn.ensemble.IsolationForest(**kwargs)
	IF.fit (X)
	if return_scores:
		return -IF.score_samples(X)
	is_inlier = IF.fit_predict(X)
	return is_inlier == -1
	
def DBSCAN(X, **kwargs):
	dbscan = sklearn.cluster.DBSCAN(**kwargs)
	clusters = dbscan.fit_predict(X)
	#if return_scores:
		#metric = kwargs['metric'] if 'metric' in kwargs else 'euclidean'
		#return distance.cdist(X, X[dbscan.core_sample_indices_,:], metric=metric).min(axis=1)
	return clusters == -1
	
	
def KNN(X, return_scores = False, **kwargs):
	knn = pyod.models.knn.KNN(**kwargs)
	is_outlier = knn.fit_predict (X)
	if return_scores:
		return knn.decision_scores_
	return is_outlier == 1
	
def ABOD(X, return_scores = False, **kwargs):
	abod = pyod.models.abod.ABOD(**kwargs)
	is_outlier = abod.fit_predict (X)
	if return_scores:
		return abod.decision_scores_  - min(abod.decision_scores_)
	return is_outlier == 1
	
def LOCI(X, return_scores = False, **kwargs):
	loci = pyod.models.loci.LOCI(**kwargs)
	is_outlier = loci.fit_predict(X)
	if return_scores:
		return loci.decision_scores_
	return is_outlier == 1

def HBOS(X, return_scores = False, **kwargs):
	hbos = pyod.models.hbos.HBOS(**kwargs)
	is_outlier = hbos.fit_predict(X)
	if return_scores:
		return hbos.decision_scores_
	return is_outlier == 1

def HDBSCAN(X, return_scores = False, **kwargs):
	h = hdbscan.HDBSCAN(**kwargs)
	clusters = h.fit_predict (X)
	if return_scores:
		return h.outlier_scores_
	return clusters == -1
	
def kmeans(X, n_clusters = 8, return_scores = False, contamination=0.1, metric='euclidean'):
	[m, n] = X.shape
	l = math.floor(m*contamination)
	indices = np.random.permutation(m)
	C = X[indices[0:n_clusters],:]
	
	while True:
		dist = distance.cdist(X,C, metric)
		nearest = np.argmin(dist,axis=1)
		dist_nearest = dist[np.arange(m),nearest]
		dist_sorted = np.argsort(dist_nearest)
		
		nearest[dist_sorted[-l:]] = -1
		
		C2 = np.empty((0,n))
		
		for j in range(n_clusters):
			indices = np.argwhere(nearest==j)
			if indices.size > 0:
				C2 = np.append(C2, np.mean(X[indices,:], axis=0), axis=0)
	
		if (C == C2).all():
			break
		C = C2

	if return_scores:
		return dist_nearest
		
	threshold = dist_nearest[dist_sorted[-l-1]]
	return dist_nearest > threshold

def SDO(X, k=None, q=None, qv=0.3, x=6, hbs=False, return_scores = False, contamination=0.1, chunksize=1000, metric='euclidean'):
	[m, n] = X.shape
	
	if hasattr(type(k), '__iter__'):
		observers = X[k]
		k = observers.shape[0]
	else:
		if k is None:
			# choose number of observers as described in paper
			pca = sklearn.decomposition.PCA()
			pca.fit(X)
			var = max(1,pca.explained_variance_[0])
			sqerr = 0.01 * pca.explained_variance_[0]
			Z = 1.96
			k = int((m * Z**2 * var) // ((m-1) * sqerr + Z**2 * var))
			
		if hbs:
			Y = X.copy()
			binning_param = 20
			for i in range(n):
				dimMin = min(Y[:,i])
				dimMax = max(Y[:,i])
				if dimMax > dimMin:
					binWidth = (dimMax - dimMin) / round(math.log10(k) * binning_param)
					Y[:,i] = np.floor( (Y[:,i] - dimMin) / binWidth) * binWidth + dimMin
			Y = np.unique(Y, axis=0)
		else:
			Y = X
			
		index = np.random.permutation(Y.shape[0])

		observers = Y[index[0:k]]
		
	# copy for efficient cache usage
	observers = observers.copy()
	
	# TRAINING
	P = np.zeros(k)

	for i in range(0,m,chunksize):
		dist = distance.cdist(X[i:(i+chunksize)], observers, metric)
		dist_sorted = np.argsort(dist, axis=1)
		closest = dist_sorted[:,0:x].flatten()

		P += np.sum (closest[:,np.newaxis] == np.arange(k), axis=0)
		
	if q is None:
		q = np.sort(P)[math.floor(k * qv)]
	
	observers = observers[P>=q].copy()
	
	# APPLICATION
	y = np.zeros(m)

	for i in range(0,m,chunksize):
		dist = distance.cdist(X[i:(i+chunksize)], observers, metric)
		dist_sorted = np.sort(dist, axis=1)
		y[i:(i+chunksize)] = np.median(dist_sorted[:,0:x], axis=1)

	if return_scores:
		return y
		
	y_sorted = np.sort(y)
	threshold = y_sorted[-math.floor(m*contamination)-1]
	
	return y > threshold

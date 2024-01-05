# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.

@author: Kevin S. Xu, Adam Neber, Saurav Giri
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import KMeans, BisectingKMeans, MeanShift, OPTICS, DBSCAN, AgglomerativeClustering, SpectralClustering

def predictWithK(testFeatures, numVessels, trainFeatures=None, 
                 trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    scaler = StandardScaler()
    norm = Normalizer()
    testFeatures = scaler.fit_transform(testFeatures)
    
    sc = SpectralClustering(n_clusters=numVessels, degree=5, eigen_solver='arpack')
    predVessels = sc.fit_predict(testFeatures)
    
    ## ac = AgglomerativeClustering(n_clusters=numVessels)
    ## predVessels = ac.fit_predict(testFeatures)

    ## db = DBSCAN()
    ## predVessels = db.fit_predict(testFeatures)
    
    ## opt = OPTICS()
    ## predVessels = opt.fit_predict(testFeatures)
    
    ## ms = MeanShift()
    ## predVessels = ms.fit_predict(testFeatures)

    ## bkm = BisectingKMeans(n_clusters=numVessels, n_init=20)
    ## predVessels = bkm.fit_predict(testFeatures)

    return predVessels

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    ## Calculate n - math is arbitrary but I found this to work pretty well
    n = round(math.sqrt(len(testFeatures)/2)/5)
    return predictWithK(testFeatures, n, trainFeatures, trainLabels)

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    
    from utils import loadData, plotVesselTracks
    data = loadData('data/set2.csv')
    features = data[:,2:]
    labels = data[:,1]

    #%% Plot all vessel tracks with no coloring
    plotVesselTracks(features[:,[2,1]])
    plt.title('All vessel tracks')
    
    #%% Run prediction algorithms and check accuracy
    
    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size
    predVesselsWithK = predictWithK(features, numVessels)
    ariWithK = adjusted_rand_score(labels, predVesselsWithK)
    
    # Prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(features)
    predNumVessels = np.unique(predVesselsWithoutK).size
    ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)
    
    print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
          + f'{ariWithoutK}')

    #%% Plot vessel tracks colored by prediction and actual labels
    plotVesselTracks(features[:,[2,1]], predVesselsWithK)
    plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features[:,[2,1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    plotVesselTracks(features[:,[2,1]], labels)
    plt.title('Vessel tracks by label')
    
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import os

import fastcluster
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn import manifold
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

from scipy import stats

def groups_in_cluster(origLabels, clusterLabels, outFn):
    # Want percentage of each cluster made up by each class
    # for each cluster
    for i in np.unique(clusterLabels):
        with open(outFn, "a") as f:
            clusterIdxs = np.where(clusterLabels == i)
            origSubset = origLabels[clusterIdxs]
            N = len(origSubset)
            f.write("Cluster, "+str(i)+", "+str(N)+"\n")
            # Get the number of elements in the cluster belonging to each original label
            f.write("Original value, count of class in cluster, percentage of cluster made up of class\n")
            for j in np.unique(origSubset):
                f.write(str(j)+", "+str(np.count_nonzero(origSubset == j))+", "+str(np.round(np.count_nonzero(origSubset == j)/float(N), 2))+"\n")

            f.write("\n")

def group_across_clusters(origLabels, clusterLabels, outFn):
    # Want percentage of each original class that belongs to each cluster
    for i in np.unique(origLabels):
        classIdxs = np.where(origLabels == i)
        if len(classIdxs) > 0:
            print(classIdxs)
            clusterSubset = clusterLabels[classIdxs]
            print("Class", i, "----------------------------------")
            # Get the number of elements in each class belonging to each cluster
            N = len(clusterSubset)
            print("Class size:", N)
            print("Cluster Number, Count of class in cluster, Percentage of class in cluster")
            for j in np.unique(clusterSubset):
                print(j, np.count_nonzero(clusterSubset == j), np.round(np.count_nonzero(clusterSubset == j)/float(N), 2))

        print()



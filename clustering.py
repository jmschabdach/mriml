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

def agglomerativeClustering(df, demoDf, metric, label, clusterRows=False, useMask=False, outFn=""):

    # Filter DFs
    metaScans = list(set(list(demoDf['ID'])))
    metricScans = list(set(list(df)))

    extraMeta = [ i for i in metaScans if i not in metricScans]
    extraMetrics = [ i for i in metricScans if i not in metaScans]

    metaScans = [i for i in metaScans if i not in extraMeta]

    demoDf = demoDf[demoDf['ID'].isin(metaScans)]
    df = df.drop(columns=extraMetrics)

    # Generate a mask
    if useMask:
        mask = np.ones_like(df, dtype=np.bool)
        locs = np.where(df != 0.0)
        for i, j in zip(locs[0], locs[1]):
            mask[i][j] = False
    else:
        mask = None

    demoDf = demoDf.drop_duplicates(subset=['ID'])

    print(demoDf.shape)
    print(df.shape)

    # Colormap for column labels
    cmap = plt.get_cmap('tab10')
    colors=[]
    for i in range(len(np.unique(demoDf[label]))):
        colors.append(cmap(i))

    lut = dict(zip(np.unique(demoDf[label]), colors))
    col_colors = demoDf[label].map(lut)
#     print(len(demoDf[label]))
#     print(len(col_colors))
#     print(col_colors.get_values())

    # Generate figure
    g = sns.clustermap(df, mask=mask, metric='minkowski', row_cluster=clusterRows,
                       col_colors=col_colors.values, cmap='plasma')

    for l in lut:
        g.ax_col_dendrogram.bar(0, 0, color=lut[l], label=l, linewidth=0)

    g.ax_col_dendrogram.legend(loc="upper center", ncol=len(np.unique(demoDf[label])))
    g.cax.set_position([.99, .2, .03, .45])

    label = label.replace("_", " ")
    g.fig.suptitle("Agglomerative Clustering of Clinical Brain Images "+metric+" by "+label,
                   x=0.6, y=1.0,
                   ha='center', va='bottom')

    if outFn != "":
        plt.savefig(outFn, bbox_inches='tight', pad_inches=0.01)


def kmeans(df, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df)
    return kmeans

def spectralClustering(df, k):
    spectral = SpectralClustering(n_clusters=k, random_state=0).fit(df)
    return spectral

def sklearnAgg(df, k):
    agg = AgglomerativeClustering(n_clusters=k).fit(df)
    return agg

def getPCAVecs(df):
    new_vecs = PCA(n_components=2, random_state=0).fit_transform(df)
    return new_vecs

def getTSNEVecs(df):
    tsne = manifold.TSNE(n_components=2, random_state=0)
    new_vecs = tsne.fit_transform(df)
    return new_vecs

def getUMAPVecs(df):
    new_vecs = umap.UMAP().fit_transform(df)
    return new_vecs

def analyzeClusterContents(origLabels, clusterLabels, outFn):
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

#    # Want percentage of each original class that belongs to each cluster
#    for i in np.unique(origLabels):
#        classIdxs = np.where(origLabels == i)
#        if len(classIdxs) > 0:
#            print(classIdxs)
#            clusterSubset = clusterLabels[classIdxs]
#            print("Class", i, "----------------------------------")
#            # Get the number of elements in each class belonging to each cluster
#            N = len(clusterSubset)
#            print("Class size:", N)
#            print("Cluster Number, Count of class in cluster, Percentage of class in cluster")
#            for j in np.unique(clusterSubset):
#                print(j, np.count_nonzero(clusterSubset == j), np.round(np.count_nonzero(clusterSubset == j)/float(N), 2))
#
#        print()



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import os

import fastcluster
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from scipy.cluster import hierarchy
from scipy.spatial import distance
from sklearn import manifold
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

from scipy import stats

## Generate column labels for the Seaborn clustermap
#  @param labels A numpy.ndarray or a list of numpy.ndarray objects containing subject labels
#  @return colors A list (or list of lists) of colors representing the subject labels 
#  @return lut A dictionary mapping unique label values to their representative colors
def generateColumnLabels(labels):

    cmap = plt.get_cmap('tab10')
    lut = {}
    colors = []

    for l in range(len(np.unique(labels))):
        lut[np.unique(labels)[l]] = cmap(l)

    if type(labels) is list:
        for lset in labels:
            colors.append([lut[i] for i in lset])

    else:
        colors = [lut[i] for i in labels]

    return colors, lut

##
#
def makeParallelPlots(df, labels, titleString, orderMagnitude=False, useMask=False, outFn=""):
    # If orderMagnitude, sort the metric values
    if orderMagnitude:
        df = pd.DataFrame({key: sorted(value.values(), reverse=True) for key, value in df.to_dict().items()})

    # Get the column colors from the labels
    colColors, legendLut = generateColumnLabels(labels)

    # Perform the clustering
    colLinkages = seabornAgg(df)

    # Graph the results on a linear scale
    g_lin = graphSeabornAgg(df, colColors, legendLut, colLinkages, titleString+" Linear Scale")

    # Graph the results on a log scale
    loggedDf = (np.log(df)) #.replace(-np.inf, 0)
    g_log = graphSeabornAgg(loggedDf, colColors, legendLut, colLinkages, titleString+" Log Scale")


##
#
def graphSeabornAgg(df, col_colors, lut, colLinks, titleString, useMask=False, outFn=""):

    # Generate a mask
    if useMask:
        mask = np.ones_like(df, dtype=np.bool)
        locs = np.where(df != -np.inf)
        for i, j in zip(locs[0], locs[1]):
            mask[i][j] = False
    else:
        mask = None

    # Colormap for data
    my_cmap = cm.get_cmap('plasma')
    my_cmap.set_bad((0,0,0))

    # Generate figure
    g = sns.clustermap(df, row_cluster=False, col_linkage=colLinks, mask=mask, 
                       col_colors=col_colors, cmap=my_cmap)

    labels = []
    for l in lut:
        g.ax_col_dendrogram.bar(0, 0, color=lut[l], label=l, linewidth=0)
        labels.append(l)

    g.ax_col_dendrogram.legend(loc="upper center", ncol=len(np.unique(labels)))
    g.cax.set_position([.99, .2, .03, .45])

    g.fig.suptitle("Agglomerative Clustering of Clinical Brain Images "+titleString,
                   x=0.6, y=1.0,
                   ha='center', va='bottom')

    if outFn != "":
        plt.savefig(outFn, bbox_inches='tight', pad_inches=0.01)

    return g


def kmeans(df, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df)
    return kmeans

def spectralClustering(df, k):
    spectral = SpectralClustering(n_clusters=k, random_state=0).fit(df)
    return spectral

def sklearnAgg(df, k):
    agg = AgglomerativeClustering(n_clusters=k).fit(df)
    return agg

def seabornAgg(df):
    colLinks = hierarchy.linkage(distance.pdist(df.T, metric="minkowski"), method="average")
    return colLinks

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



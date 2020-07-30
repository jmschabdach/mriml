import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import os

from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn import manifold
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap


def kmeansClustering(df, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df)
    return kmeans

def spectralClustering(df, k):
    spectral = SpectralClustering(n_clusters=k, random_state=0).fit(df)
    return spectral

def agglomerativeClustering(df, k):
    agg = AgglomerativeClustering(n_clusters=k).fit(df)
    return agg

def agglomerativeClusteringSns(df, labels, colors, outFn=""):

    lookuptable = dict(zip(siteIds.unique(), colors))
    row_colors = siteIds.map(lookuptable)
    legHandles = [Patch(color=lookuptable[i], label=i) for i in lookuptable]
    
    clustmap = sns.clustermap(df.T, metric="cosine", cmap='mako', row_cluster=False, col_colors=row_colors)
    leg = clustmap.ax_heatmap.legend(handles=legHandles, bbox_to_anchor=(1.3, 0.9))
    leg.set_title(title="Subject Labels")
        
#     plt.colorbar(plt.cm.ScalarMappable(cmap='mako'))
        
    if outFn != "":
        print("save")

def analyzeClusterContents(origLabels, clusterLabels):
    # Want percentage of each cluster made up by each class
    # for each cluster
    for i in np.unique(clusterLabels):
        clusterIdxs = np.where(clusterLabels == i)
        origSubset = origLabels[clusterIdxs]
        print("Cluster", i, "--------------------------------")
        # Get the number of elements in the cluster belonging to each original label
        N = len(origSubset)
        print("Cluster size:", N)
        print("Original value, count of class in cluster, percentage of cluster made up of class")
        for j in np.unique(origSubset):
            print(j, np.count_nonzero(origSubset == j), np.round(np.count_nonzero(origSubset == j)/float(N), 2))
            
        print()
        
    # Want percentage of each original class that belongs to each cluster
    for i in np.unique(origLabels):
        classIdxs = np.where(origLabels == i)
        clusterSubset = clusterLabels[classIdxs]
        print("Class", i, "----------------------------------")
        # Get the number of elements in each class belonging to each cluster
        N = len(clusterSubset)
        print("Class size:", N)
        print("Cluster Number, Count of class in cluster, Percentage of class in cluster")
        for j in np.unique(clusterSubset):
            print(j, np.count_nonzero(clusterSubset == j), np.round(np.count_nonzero(clusterSubset == j)/float(N), 2))
            
        print()
            
            

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



def plotClusters(data, clusterLabels, origLabels, colors, title="", outFn=""):
    # Get the lower dimensional vectors
    pcaVecs = getPCAVecs(data)
    tsneVecs = getTSNEVecs(data)
    umapVecs = getUMAPVecs(data)
    
    # get colors
    origLut = dict(zip(origLabels.unique(), colors))
    origColors = origLabels.map(origLut)

    clusterLut = dict(zip(np.unique(clusterLabels), colors))
    clusterColors = [clusterLut[i] for i in clusterLabels]
    
    # Create legend elements
    origLegEl = [Line2D([0], [0], marker='o', color=origLut[i], label=i) for i in origLut.keys()]
    
    clusterLegEl = [Line2D([0], [0], marker='o', color=clusterLut[i], label="Cluster "+str(i)) for i in clusterLut.keys()]

    # Create a figure
    f, axs = plt.subplots(3, 2, figsize=(10, 15))
    
    # Add 6 subplots (3 row, 2 col)
    scatter = axs[0, 0].scatter(pcaVecs[:,0], pcaVecs[:,1], c=origColors)
    axs[0,0].set_title("PCA with Original Labels")
    axs[0,0].legend(handles=origLegEl, bbox_to_anchor=(0, 1.22, 1., 0.122 ),
                    loc='upper left',  ncol=2, mode="expand")

    plt.subplot(3, 2, 2)
    scatter = plt.scatter(pcaVecs[:,0], pcaVecs[:,1], c=clusterColors)
    plt.title("PCA with Cluster Labels")
    plt.legend(handles=clusterLegEl, bbox_to_anchor=(0, 1.22, 1., 0.122 ),
                    loc='upper left',  ncol=2, mode="expand")
    
    plt.subplot(3, 2, 3)
    plt.scatter(tsneVecs[:,0], tsneVecs[:,1], c=origColors)
    plt.title("t-SNE with Original Labels")
    plt.subplot(3, 2, 4)
    plt.scatter(tsneVecs[:,0], tsneVecs[:,1], c=clusterColors)
    plt.title("t-SNE with Cluster Labels")
    
    plt.subplot(3, 2, 5)
    plt.scatter(umapVecs[:,0], umapVecs[:,1], c=origColors)
    plt.title("UMAP with Original Labels")
    plt.subplot(3, 2, 6)
    plt.scatter(umapVecs[:,0], umapVecs[:,1], c=clusterColors)
    plt.title("UMAP with Cluster Labels")
    
    # show the image
    plt.suptitle(title, fontsize=16)
    
    plt.show()
    
    # set up for save
    if outFn != "":
        print("save")


def plotClustersSameNAllMethods(data, origLabels, kmeansLabels, spectralLabels, agglomLabels, colors, title="", outFn=""):
    # Get the lower dimensional vectors
    pcaVecs = getPCAVecs(data)
    tsneVecs = getTSNEVecs(data)
    umapVecs = getUMAPVecs(data)
    
    # get colors
    origLut = dict(zip(origLabels.unique(), colors))
    origColors = origLabels.map(origLut)

    kmeansLut = dict(zip(np.unique(kmeansLabels), colors))
    kmeansColors = [kmeansLut[i] for i in kmeansLabels]
    
    spectralLut = dict(zip(np.unique(spectralLabels), colors))
    spectralColors = [spectralLut[i] for i in spectralLabels]
    
    agglomLut = dict(zip(np.unique(agglomLabels), colors))
    agglomColors = [agglomLut[i] for i in agglomLabels]
    
    # Create legend elements
    origLegEl = [Line2D([0], [0], marker='o', color=origLut[i], label=i) for i in origLut.keys()]
    
    kmeansLegEl = [Line2D([0], [0], marker='o', color=kmeansLut[i], label="Cluster "+str(i)) for i in kmeansLut.keys()]
    spectralLegEl = [Line2D([0], [0], marker='o', color=spectralLut[i], label="Cluster "+str(i)) for i in spectralLut.keys()]
    agglomLegEl = [Line2D([0], [0], marker='o', color=agglomLut[i], label="Cluster "+str(i)) for i in agglomLut.keys()]


    # Create a figure
    f, axs = plt.subplots(3, 4, figsize=(20, 15))
    
    # Add 6 subplots (3 row, 2 col)
    plt.subplot(3,4,1)
    scatter = plt.scatter(pcaVecs[:,0], pcaVecs[:,1], c=origColors)
    plt.title("PCA with Original Labels")
    plt.legend(handles=origLegEl, bbox_to_anchor=(0, 1.22, 1., 0.122 ),
                    loc='upper left',  ncol=2, mode="expand")

    plt.subplot(3, 4, 2)
    scatter = plt.scatter(pcaVecs[:,0], pcaVecs[:,1], c=kmeansColors)
    plt.title("PCA with Kmeans Labels")
    plt.legend(handles=kmeansLegEl, bbox_to_anchor=(0, 1.22, 1., 0.122 ),
                    loc='upper left',  ncol=2, mode="expand")
    
    plt.subplot(3, 4, 3)
    scatter = plt.scatter(pcaVecs[:,0], pcaVecs[:,1], c=spectralColors)
    plt.title("PCA with Spectral Labels")
    plt.legend(handles=spectralLegEl, bbox_to_anchor=(0, 1.22, 1., 0.122 ),
                    loc='upper left',  ncol=2, mode="expand")
    
    plt.subplot(3, 4, 4)
    scatter = plt.scatter(pcaVecs[:,0], pcaVecs[:,1], c=agglomColors)
    plt.title("PCA with Agglomerative Labels")
    plt.legend(handles=agglomLegEl, bbox_to_anchor=(0, 1.22, 1., 0.122 ),
                    loc='upper left',  ncol=2, mode="expand")
    
    plt.subplot(3, 4, 5)
    plt.scatter(tsneVecs[:,0], tsneVecs[:,1], c=origColors)
    plt.title("t-SNE with Original Labels")
    plt.subplot(3, 4, 6)
    plt.scatter(tsneVecs[:,0], tsneVecs[:,1], c=kmeansColors)
    plt.title("t-SNE with Kmeans Labels")
    plt.subplot(3, 4, 7)
    plt.scatter(tsneVecs[:,0], tsneVecs[:,1], c=spectralColors)
    plt.title("t-SNE with Spectral Labels")
    plt.subplot(3, 4, 8)
    plt.scatter(tsneVecs[:,0], tsneVecs[:,1], c=agglomColors)
    plt.title("t-SNE with Agglomerative Labels")
    
    plt.subplot(3, 4, 9)
    plt.scatter(umapVecs[:,0], umapVecs[:,1], c=origColors)
    plt.title("UMAP with Original Labels")
    plt.subplot(3, 4, 10)
    plt.scatter(umapVecs[:,0], umapVecs[:,1], c=kmeansColors)
    plt.title("UMAP with Kmeans Labels")
    plt.subplot(3, 4, 11)
    plt.scatter(umapVecs[:,0], umapVecs[:,1], c=spectralColors)
    plt.title("UMAP with Spectral Labels")
    plt.subplot(3, 4, 12)
    plt.scatter(umapVecs[:,0], umapVecs[:,1], c=agglomColors)
    plt.title("UMAP with Agglomerative Labels")
    
    # show the image
    plt.suptitle(title, fontsize=16)
    
    plt.show()
    
    # set up for save
    if outFn != "":
        print("save")
    

def plotClustersSameMethodDifferentN(data, origLabels, cluster1Labels, cluster2Labels, cluster3Labels, colors, title="", outFn=""):
    # Get the lower dimensional vectors
    pcaVecs = getPCAVecs(data)
    tsneVecs = getTSNEVecs(data)
    umapVecs = getUMAPVecs(data)
    
    # get colors
    origLut = dict(zip(origLabels.unique(), colors))
    origColors = origLabels.map(origLut)

    cluster1Lut = dict(zip(np.unique(cluster1Labels), colors))
    cluster1Colors = [cluster1Lut[i] for i in cluster1Labels]
    
    cluster2Lut = dict(zip(np.unique(cluster2Labels), colors))
    cluster2Colors = [cluster2Lut[i] for i in cluster2Labels]
    
    cluster3Lut = dict(zip(np.unique(cluster3Labels), colors))
    cluster3Colors = [cluster3Lut[i] for i in cluster3Labels]
    
    # Create legend elements
    origLegEl = [Line2D([0], [0], marker='o', color=origLut[i], label=i) for i in origLut.keys()]
    
    cluster1LegEl = [Line2D([0], [0], marker='o', color=cluster1Lut[i], label="Cluster "+str(i)) for i in cluster1Lut.keys()]
    cluster2LegEl = [Line2D([0], [0], marker='o', color=cluster2Lut[i], label="Cluster "+str(i)) for i in cluster2Lut.keys()]
    cluster3LegEl = [Line2D([0], [0], marker='o', color=cluster3Lut[i], label="Cluster "+str(i)) for i in cluster3Lut.keys()]


    # Create a figure
    f, axs = plt.subplots(3, 4, figsize=(20, 15))
    
    # Add 6 subplots (3 row, 2 col)
    plt.subplot(3,4,1)
    scatter = plt.scatter(pcaVecs[:,0], pcaVecs[:,1], c=origColors)
    plt.title("PCA with Original Labels")
    plt.legend(handles=origLegEl, bbox_to_anchor=(0, 1.22, 1., 0.122 ),
                    loc='upper left',  ncol=2, mode="expand")

    plt.subplot(3, 4, 2)
    scatter = plt.scatter(pcaVecs[:,0], pcaVecs[:,1], c=cluster1Colors)
    plt.title("PCA with "+str(len(cluster1Lut.keys()))+" Clusters")
    plt.legend(handles=cluster1LegEl, bbox_to_anchor=(0, 1.22, 1., 0.122 ),
                    loc='upper left',  ncol=2, mode="expand")
    
    plt.subplot(3, 4, 3)
    scatter = plt.scatter(pcaVecs[:,0], pcaVecs[:,1], c=cluster2Colors)
    plt.title("PCA with "+str(len(cluster2Lut.keys()))+" Clusters")
    plt.legend(handles=cluster2LegEl, bbox_to_anchor=(0, 1.22, 1., 0.122 ),
                    loc='upper left',  ncol=2, mode="expand")
    
    plt.subplot(3, 4, 4)
    scatter = plt.scatter(pcaVecs[:,0], pcaVecs[:,1], c=cluster3Colors)
    plt.title("PCA with "+str(len(cluster3Lut.keys()))+" Clusters")
    plt.legend(handles=cluster3LegEl, bbox_to_anchor=(0, 1.22, 1., 0.122 ),
                    loc='upper left',  ncol=2, mode="expand")
    
    plt.subplot(3, 4, 5)
    plt.scatter(tsneVecs[:,0], tsneVecs[:,1], c=origColors)
    plt.title("t-SNE with Original Labels")
    plt.subplot(3, 4, 6)
    plt.scatter(tsneVecs[:,0], tsneVecs[:,1], c=cluster1Colors)
    plt.title("t-SNE with "+str(len(cluster1Lut.keys()))+" Clusters")
    plt.subplot(3, 4, 7)
    plt.scatter(tsneVecs[:,0], tsneVecs[:,1], c=cluster2Colors)
    plt.title("t-SNE with "+str(len(cluster2Lut.keys()))+" Clusters")
    plt.subplot(3, 4, 8)
    plt.scatter(tsneVecs[:,0], tsneVecs[:,1], c=cluster3Colors)
    plt.title("t-SNE with "+str(len(cluster3Lut.keys()))+" Clusters")
    
    plt.subplot(3, 4, 9)
    plt.scatter(umapVecs[:,0], umapVecs[:,1], c=origColors)
    plt.title("UMAP with Original Labels")
    plt.subplot(3, 4, 10)
    plt.scatter(umapVecs[:,0], umapVecs[:,1], c=cluster1Colors)
    plt.title("UMAP with "+str(len(cluster1Lut.keys()))+" Clusters")
    plt.subplot(3, 4, 11)
    plt.scatter(umapVecs[:,0], umapVecs[:,1], c=cluster2Colors)
    plt.title("UMAP with "+str(len(cluster2Lut.keys()))+" Clusters")
    plt.subplot(3, 4, 12)
    plt.scatter(umapVecs[:,0], umapVecs[:,1], c=cluster3Colors)
    plt.title("UMAP with "+str(len(cluster3Lut.keys()))+" Clusters")
    
    # show the image
    plt.suptitle(title, fontsize=16)
    
    plt.show()
    
    # set up for save
    if outFn != "":
        print("save")
    
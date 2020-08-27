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
from sklearn.metrics import silhouette_score
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
def makeNormAndLogPlots(df, labels, titleString, orderMagnitude=False, useMask=False, outFn=""):
    # If orderMagnitude, sort the metric values
    if orderMagnitude:
        df = pd.DataFrame({key: sorted(value.values(), reverse=True) for key, value in df.to_dict().items()})

    # Get the column colors from the labels
    colColors, legendLut = generateColumnLabels(labels)

    # Perform the clustering
    dists = calculateDistance(df)
    colLinkages, results = agglomerative(dists, k=3)

    # Graph the results on a linear scale
    g_lin = agglomerativeClustermap(df, colColors, legendLut, colLinkages, titleString+" Linear Scale", outFn+"_linear_scale.png")

    # Graph the results on a log scale
    loggedDf = (np.log(df)) #.replace(-np.inf, 0)
    g_log = agglomerativeClustermap(loggedDf, colColors, legendLut, colLinkages, titleString+" Log Scale", outFn+"_log_scale.png")


##
#
def agglomerativeClustermap(df, col_colors, lut, colLinks, titleString, useMask=False, outFn=""):

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

def calculateDistance(df):
    # Calculate the distance matrix
    dists = distance.pdist(df.T, metric="cosine")
    return dists 

## Compute the linkages between data samples, then cluster using agglomerative clustering
#  @param df Dataframe object of shape nxm where n is the number of samples and m is the number of features
#  @param k Int number of clusters
def agglomerative(df, k):
    print("In mriml.clustering.agglomerative")
    print(df.shape)
    # Calculate the linkages between the clusters
    dists = calculateDistance(df.T)
    print(distance.squareform(dists).shape)
    links = hierarchy.linkage(np.nan_to_num(distance.squareform(dists)), method="complete")
    # Create the clustering model
    agg = AgglomerativeClustering(n_clusters=k, 
                                  affinity='precomputed',  # use the distance matrix
                                  linkage='complete')      # max of all distances between all observations of 2 sets

    # Perform clustering and return the cluster labels for the data points
    clusterResults = agg.fit_predict(np.nan_to_num(distance.squareform(dists)))
    
    return links, clusterResults

def kmeans(df, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit_predict(df)
    return kmeans

def spectral(df, k):
    spectral = SpectralClustering(n_clusters=k, random_state=0).fit_predict(df)
    return spectral

## Identify the optimal number of clusters for the set of features
#  @param features Dataframe of features being clustered
#  @param clusterAlg String specifying clustering algorithm
#  @param minK Int minumum number of clusters to test
#  @param maxK Int maximum number of clusters to test
#  @return sscores Dictionary of cluster numbers and their silhouette scores
def identifyOptimalK(features, clusterAlg, minK=2, maxK=9):
    print("In mriml.clustering.identifyOptimalK")
    print(features.shape)
    # Specify the number of clusters to test
    possibleK = list(range(minK, maxK))

    # Set up the silouette scores list
    sscores = []
    maxScore = 0
    bestK = 2

    # Test different numbers of clusters
    for k in possibleK:
        if clusterAlg == "K-means":
            results = kmeans(features, k)
        elif clusterAlg == "Spectral":
            results = spectral(features, k)
        elif clusterAlg == "Agglomerative":
            _, results = agglomerative(features, k)


        print("Clusters:", k)
        print("Number of labels:", np.unique(results))

        try:
            score = silhouette_score(features, results)

        except ValueError:
            score = 0

        sscores.append([k, score])

        if score > maxScore:
            maxScore = score
            bestK = k

        print(k, score)

        sscoresDf = pd.DataFrame(sscores, columns=['k', 'silhouette_score'])

    return sscoresDf, bestK

## Identify the optimal cluster size and perform clustering of specified type on given features
#  @param features Dataframe of one type of metrics
#  @param clusterType String specifying the type of clutering to use; currently support K-means or Spectral
#  @return clusterLabels List of labels
#  @return sscoresDf Dataframe of silhouette scores
def clusterOptimalK(features, clusterType):

    # Identify the optimal number of clusters for the set of features and clustering type
    sscoresDf, k = identifyOptimalK(features, clusterType)
    # Could save the results/score/k into a table...

    # Perform clustering on optimal cluster size
    if clusterType == 'K-means':
        clusterLabels = kmeans(features, k)
    elif clusterType == 'Spectral':
        clusterLabels = spectral(features, k)
    elif clusterType == 'Agglomerative':
        _, clusterLabels = agglomerative(features, k)

    return clusterLabels, sscoresDf

## Plot the results of a single clustering analysis using a scatterplot
#  @param features Dataframe of one type of metric
#  @param clusterLabels List of labels 
#  @param title String specifying the title associated with each metric
#  @param population String specifying the population (title purposes)
#  @param clusterType String specifying the type of clutering to use; currently support K-means or Spectral
#  @param outPath String specifying the directory to save the figures to
def scatterplotClusters(features, clusterLabels, title, population, clusterType, outPath):
    print("In mriml.clustering.scatterplotClusters")
    print(features.shape)
    print(len(clusterLabels))

    cmap = plt.get_cmap('tab10')
    
    # Define the color map
    clusterColors, lut = generateColumnLabels(clusterLabels)

    # Reduce the dimensionality of the features for graphing purposes
    pca = getPCAVecs(features)
    tsne = getTSNEVecs(features)
    umap = getUMAPVecs(features)

    # Graph the clustering results
    plt.figure(figsize=(15,5))
    plt.subplot(1, 3, 1)
    plt.scatter(pca[:,0], pca[:,1], c=clusterColors)
    plt.title('PCA')
    plt.subplot(1, 3, 2)
    plt.scatter(tsne[:,0], tsne[:,1], c=clusterColors)
    plt.title('tSNE')
    plt.subplot(1, 3, 3)
    plt.scatter(umap[:,0], umap[:,1], c=clusterColors)
    plt.title('UMAP')

    # Add the title for the set of subplots
    plt.suptitle(population+' '+title+' '+clusterType+' (k='+str(len(np.unique(clusterLabels)))+')')

    # Add the legend
    legendElements = [Line2D([0], [0], marker='o', color=lut[i], label='Cluster '+str(i+1)) for i in lut.keys()]
    plt.legend(handles=legendElements, bbox_to_anchor=(0, 1.22, 1., 0.122),
               loc='upper left', ncol=2, mode='expand')

    # Show the plot
#    plt.show()

    # Build the filename to save the figure to
    outFn = outPath + clusterType.lower().replace('-','')+'_'           # Start the output file name with the type of clustering
    outFn += population.lower()+'_'                                     # Add the age group population
    title = title.lower().translate(str.maketrans('','','$\()'))        # Replace non-alpha characters
    title = title.replace(' vector', 'vec').replace(' matrix', 'mat')   # Replace longer MI descriptions
    outFn += title.split(' ')[1]+'_'+title.split(' ')[0]+'.png'         # Add the information from the reformatted title string

    plt.savefig(outFn)

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

## Count how many samples from each class are present in each cluster
#  @param origLabels
#  @param clusterLabels
#  @param outFn
def analyzeClusterContents(origLabels, clusterLabels, outFn):
    # Want counts of each class in each cluster

    cols = ['cluster']+list(np.unique(origLabels))
    rows = []

    # for each cluster
    for i in np.unique(clusterLabels):
        # get the indices belonging to that cluster
        clusterIdxs = np.where(clusterLabels == i)
        # get the original labels of the items in the cluster
        origSubset = origLabels[clusterIdxs]
        # get the number of occurences of each label in the cluster
        counts = [np.count_nonzero(origSubset == j) for j in np.unique(origLabels)]
        rows.append([i]+counts)


    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(outFn)

    print(df)


## Calculate the intra and inter cluster similarities
#  @param
def analyzeClusterSimilarities():
    pass

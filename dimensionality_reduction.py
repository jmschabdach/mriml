from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

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


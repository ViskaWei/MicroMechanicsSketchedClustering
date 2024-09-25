import os
import numpy as np
import pandas as pd
import seaborn as sns
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

########################### Loading #######################################

def load_data(DATASET, index_col_name="Number"):
    if DATASET[-3:]=='csv':
        data = pd.read_csv(DATASET, delimiter=",", index_col=index_col_name)     
    elif DATASET[-4:]=='xlsx':
        data = pd.read_excel(DATASET, index_col=index_col_name)    
    else:
        raise 'can only read csv or xlsx file'
    return data

def prepro_data(data, drop=True, isCenter=False):
    vol = data.pop('vol')
    if drop:
        cols = data.columns[(data.sum()!=0)]
        data = data[cols]
    else:
        cols = data.columns
    if isCenter: data = data - data.mean().mean() 
    return data, cols, vol    

########################### Preprocessing #######################################

def get_pca(mat, dim=6):
    pca = PCA(n_components=dim, random_state = 227)    
    matPCA=pca.fit_transform(mat)    
    print(matPCA.shape)
    return matPCA

def get_umap(matPCA):
    umapT = umap.UMAP(n_components=2, min_dist=0.0, n_neighbors=50, random_state=227)
    matUMAP = umapT.fit_transform(matPCA)
    print(matUMAP.shape)
    return matUMAP

def get_cluster(outEmbed, nCluster):
    kmap = KMeans(n_clusters=nCluster,n_init=30, algorithm='elkan',random_state=227)
    kmap.fit(outEmbed, sample_weight = None)
    cluster_id = kmap.labels_ + 1
    min_dist = np.min(cdist(outEmbed, kmap.cluster_centers_, 'euclidean'), axis=1)    
    return cluster_id, min_dist, kmap

########################### Plotting #######################################

def plot_data(data,kmap, cut = 2000, rng=50):
    f, axes = plt.subplots(2,2, figsize=(16,10))
    sns.scatterplot(ax=axes[0][0],
            x='t1', y='t2',
            hue= kmap.labels_+1 , marker='x',s=5,
            palette=sns.color_palette("muted", kmap.n_clusters),
            data=data,
            legend="full")
    axes[0][0].scatter(kmap.cluster_centers_[:,0],kmap.cluster_centers_[:,1], c='r') 
    axes[0][1].scatter(list(range(data.shape[0])), data[f'C{kmap.n_clusters}'])
    axes[1][1].scatter(list(range(data.shape[0])), data[f'C{kmap.n_clusters}'])
    axes[1][1].axvline(cut)
    axes[1][1].set_xlim(cut-rng,cut+rng)

def plot_umap(data, kmap):
    f, ax = plt.subplots(1, figsize=(10,8))
    sns.scatterplot(ax=ax,
            x='t1', y='t2',
            hue= kmap.labels_+1 , marker='x',s=5,
            palette=sns.color_palette("muted", kmap.n_clusters),
            data=data,
            legend="full")

    ax.scatter(kmap.cluster_centers_[:,0],kmap.cluster_centers_[:,1], c='r') 

########################### Saving #######################################
def save_cluster_ids(data, nCluster, outDir=None,name='kMat'):
    if outDir is None: outDir = './'
    if os.path.exists(outDir) is False: os.mkdir(outDir)
    lbl = data[f'C{nCluster}']
    for i in range(1,nCluster+1):
        c = list(data[lbl == i].index)
        print(f'Cluster{i} of len{len(c)}: {c[:3]}..')
        np.savetxt(f'{outDir}{name}_C{nCluster}_c{i}.txt', c, fmt="%d")     

def save_centers(cMat,nCluster,outDir=None,name='cMat'):
    if outDir is None: outDir = './'
    cMat.to_csv(f'{outDir}{name}_C{nCluster}.csv') 

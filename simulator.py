from wsi_mil.tile_wsi.sampler import TileSampler
import copy
import openslide
import pickle
from wsi_mil.tile_wsi.utils import get_image
import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, clone
from sklearn.utils.metaestimators import if_delegate_has_method
import einops
from PIL import Image
import os
from glob import glob
from argparse import Namespace
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import pdist, squareform

class InductiveClusterer(BaseEstimator):
    """
    Code from https://scikit-learn.org/stable/auto_examples/cluster/plot_inductive_clustering.html
    """
    def __init__(self, clusterer, classifier):
        self.clusterer = clusterer
        self.classifier = classifier

    def fit(self, X, y=None):
        self.clusterer_ = clone(self.clusterer)
        self.classifier_ = clone(self.classifier)
        y = self.clusterer_.fit_predict(X)
        self.classifier_.fit(X, y)
        return self

    @if_delegate_has_method(delegate='classifier_')
    def predict(self, X):
        return self.classifier_.predict(X)

    @if_delegate_has_method(delegate='classifier_')
    def decision_function(self, X):
        return self.classifier_.decision_function(X)

class DatasetClusterer():
    """
    Clusters a whole dataset of tile. ~millions of tiles.
    """
    def __init__(self, moco_path, n_cluster=10):
        self.moco_path = moco_path
        self.n_cluster = n_cluster
        self.size_emb = 512
        self.info_path = os.path.join(moco_path, 'info')
        self.mat_path = os.path.join(moco_path, 'mat_pca')
        self.clusterer = InductiveClusterer(KMeans(n_cluster), SVC(kernel='rbf', class_weight='balanced'))
        self.train_classifier(n_dim=10)
        self.preds = self.cluster_dataset(n_dim=10)
        self.counter = self.count(self.preds)
        self.total_t = np.sum([n for n in self.counter.items()])
        self.centers = self.compute_cluster_centers(self.preds)
        self.pools = self.make_pool_for_sampling(self.preds)

    def train_classifier(self, n_dim=5):
        args = Namespace(sampler='dpp', nb_tiles=20)
        s = []
        for wsi in glob(os.path.join(self.mat_path, '*.npy')):
            m = np.load(wsi)
            ts = TileSampler(args=args, wsi_path=wsi, info_folder=self.info_path)
            s.append(m[ts.dpp_sampler(20),:])
        self.clusterer.fit(np.vstack(s)[:,:n_dim])
    
    def _get_ID_from_path(self, path):
        return os.path.basename(path).split('_')[0]

    def cluster_dataset(self, n_dim=5):
        preds = dict()
        for wsi in glob(os.path.join(self.mat_path, '*.npy')):
            m = np.load(wsi)
            preds[wsi] = self.clusterer.predict(m[:,:n_dim]).reshape(-1,1)
        return preds

    def count(self, preds):
        return Counter(np.vstack([preds[x] for x in preds]).reshape(-1))
    
    def compute_cluster_centers(self, preds):
        centers=dict()
        for i in range(self.n_cluster):
            centers[i] = np.zeros(self.size_emb)
        for k in preds:
            m = np.load(k)
            for label in range(self.n_cluster):
                indices = np.where(preds[k].reshape(-1) == label)[0]
                centers[label] += m[indices, :].sum(axis=0) if len(indices) else 0
        for label in range(self.n_cluster):
            centers[label] /= self.counter[label]
        return centers
    
    def make_pool_for_sampling(self, preds):
        pools = dict()
        for label in range(self.n_cluster):
            pools[label] = []
        for wsi in preds:
            for o,p in enumerate(preds[wsi]):
                pools[p.item()].append((o, self._get_ID_from_path(wsi)))
        pools = {k:np.array(e) for k,e in pools.items()}
        return pools
            
    def compute_centers_statistics(self, centers):
        """
        computes distances between centers.
        """
        c = [centers[x] for x in range(self.n_cluster)]
        c = np.vstack(c)
        dist = squareform(pdist(c))
        return dist

class WSISimulator(DatasetClusterer):
    def __init__(self, moco_path, emb_path, n_cluster=10):
        """
        emb_path is the path for the embeddings to use for creating the dataset.
        """
        super(WSISimulator, self).__init__(moco_path, n_cluster)
        self.emb_path = emb_path
        self.pools_c = copy.deepcopy(self.pools)

    def extract_images(self, cluster, raw_path, out_path):
        """
        Just to have a look at the clusters.
        """
        ims = {k:[] for k in self.pools.keys()}
        for l in self.pools.keys():
            selection = self.pools[l][np.random.choice(len(self.pools[l]), size=100, replace=False)]
            for s in selection:
                with open(self.get_path_from_ID(s[1], 'info'), 'rb') as f:
                    info = pickle.load(f)
                slide = openslide.open_slide(glob(os.path.join(raw_path, s[1]+'*'))[0])
                ims[l].append(get_image(slide, info[int(s[0])]))
            
        ims = {k:Image.fromarray(einops.rearrange(ims[k], '(b1 b2) w h c -> (b1 w) (b2 h) c', b1=10, b2=10)) for k in ims.keys()}
        return ims

    def get_path_from_ID(self, i, file_type):
        suffix = '_infodict.pickle' if file_type == 'info' else '_embedded.npy'
        return os.path.join(self.emb_path, file_type, i+suffix)

    def sample_wsi(self, ntiles, problem, cluster, classif, prop=None):
        """presence_single_pattern.
        samples a WSI. The class 0 will be associated with the absence of the pattern
        of $cluster. 
        break down in several functions.

        :param ntiles: int, number of tiles for a WSI
        :param cluster: int, number of the class causing cluster.
        :param classif: int, 0 or 1
        """
        probas = getattr(self, f"_sample_{problem}")(cluster, classif, prop)
        sel_clust = np.random.choice(np.arange(self.n_cluster), size=ntiles, replace=False, p=probas)
        wsi = []
        composition = Counter(sel_clust)
        for c in composition.keys():
            pool = self.pools_c[c]
            ids = np.random.choice(np.arange(len(pool)),size=composition[c], replace=False)
            wsi += [np.load(self.get_path_from_ID(x[1]))[x[0], :] for x in pool[ids]]
        wsi = np.random.shuffle(wsi)
        return np.vstack(wsi)

    def _sample_presence_single_pattern(self, cluster, classif, prop):
        probas = np.zeros(len(self.n_cluster))        
        for l in self.counter.keys():
            probas[l] = self.counter[l]
        if classif == 0:
            probas[cluster] = 0
        if classif == 1:
            if prop is not None:
                n = int((probas.sum() - probas[cluster]) * prop)
                probas[cluster] = n
        probas /= probas.sum()
        return probas

    def _sample_proportion_single_pattern(self, cluster, classif, prop):
        """_sample_proportion_single_pattern.

        :param cluster: number of the cluster to enrich (or apauvrir)
        :param classif: label of the simulated slide
        :param prop: here, prop is the factor of difference between the two labels wrt the $cluster.
        i.e a slide labeled 1 will have 2 times more tiles from cluster $cluster.
        """
        prop = 2 if prop is None else prop
        probas = np.zeros(len(self.n_cluster))        
        for l in self.counter.keys():
            probas[l] = self.counter[l]
        if classif:
            probas[cluster] *= prop
        probas /= probas.sum()
        return probas


 


            

             




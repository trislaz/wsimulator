from wsi_mil.tile_wsi.sampler import TileSampler
import pandas as pd
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
        self.size_emb = 2048
        self.info_path = os.path.join(moco_path, 'info')
        self.mat_path = os.path.join(moco_path, 'mat_pca')
        self.clusterer = InductiveClusterer(KMeans(n_cluster), SVC(kernel='rbf', class_weight='balanced'))
        self.train_classifier(n_dim=20)
        self.preds = self.cluster_dataset(n_dim=20)
        self.counter = self.count(self.preds)
        self.total_t = np.sum([n for n in self.counter.items()])
        self.centers = self.compute_cluster_centers(self.preds)
        self.pools = self.make_pool_for_sampling(self.preds)

    def train_classifier(self, n_dim=5):
        args = Namespace(sampler='dpp', nb_tiles=10)
        s = []
        for wsi in glob(os.path.join(self.mat_path, '*.npy')):
            m = np.load(wsi)
            ts = TileSampler(args=args, wsi_path=wsi, info_folder=self.info_path)
            sample = ts.dpp_sampler(10)
            s.append(m[sample,:])
            print(len(sample))
        self.clusterer.fit(np.vstack(s)[:,:n_dim])
    
    def _get_ID_from_path(self, path):
        return os.path.splitext(os.path.basename(path))[0].split('_embedded')[0]

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
                print(self._get_ID_from_path(wsi))
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
    """
    TODO : quand on crée dataset : créer readme avec les caractéristiques du dataset.
    TODO : quand on crée le dataset : créer infos avec les infodict.pickle + Counter - comptant les classes.

    """
    def __init__(self, moco_path, emb_path, n_cluster=10):
        """
        emb_path is the path for the embeddings to use for creating the dataset.
        """
        super(WSISimulator, self).__init__(moco_path, n_cluster)
        self.emb_path = emb_path
        self.pools_c = copy.deepcopy(self.pools)
        self.prop = None

    def extract_images(self, raw_path, out_path):
        """
        Just to have a look at the clusters.
        """
        ims = {k:[] for k in self.pools.keys()}
        for l in self.pools.keys():
            selection = self.pools[l][np.random.choice(len(self.pools[l]), size=100, replace=False)]
            for s in selection:
                print(s)
                with open(self.get_path_from_ID(s[1], 'info'), 'rb') as f:
                    info = pickle.load(f)
                slide = openslide.open_slide(glob(os.path.join(raw_path, s[1]+'*'))[0])
                ims[l].append(get_image(slide, info[int(s[0])]))
        ims = {k:Image.fromarray(einops.rearrange(ims[k], '(b1 b2) w h c -> (b1 w) (b2 h) c', b1=10, b2=10)) for k in ims.keys()}
        [i.save(os.path.join(out_path, f'cluster_{o}.png')) for o, i in enumerate(ims)]
        return ims

    def get_path_from_ID(self, i, file_type='mat_pca'):
        suffix = '_infodict.pickle' if file_type == 'info' else '_embedded.npy'
        return os.path.join(self.emb_path, file_type, i+suffix)

    def make_dataset(self, out, nwsi, problem, cluster, prop=None):
        dic = []
        out_mat = os.path.join(out, 'mat_pca')
        out_info = os.path.join(out, 'info')
        os.makedirs(out_mat, exist_ok=True)
        os.makedirs(out_info, exist_ok=True)
        for i in range(nwsi):
            wsi, info, composition = self.sample_wsi(1000, problem, cluster, i%2, prop)
            dic.append({'ID':f'wsi_{i}', 'target': i%2})
            np.save(os.path.join(out_mat, f'wsi_{i}_embedded.npy'), wsi)
            with open(os.path.join(out_info, f'wsi_{i}_infodict.pickle'), 'wb') as f:
                pickle.dump(info, f)
            with open(os.path.join(out_info, f'wsi_{i}_compo.pickle'), 'wb') as f:
                pickle.dump(composition, f)
        table_data = pd.DataFrame(dic)
        table_data.to_csv(os.path.join(out, 'table_data.csv'), index=False)

    def write_one_wsi(self, n, problem, cluster, prop=None, ntiles=1000, out="."):
        wsi, info, composition = self.sample_wsi(ntiles, problem, cluster, n%2, prop)
        np.save(os.path.join(out, f'wsi_{n}_embedded.npy'), wsi)
        with open(os.path.join(out, f'wsi_{n}_infodict.pickle'), 'wb') as f:
            pickle.dump(info, f)
        with open(os.path.join(out, f'wsi_{n}_compo.pickle'), 'wb') as f:
            pickle.dump(composition, f)
         
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
        sel_clust = np.random.choice(np.arange(self.n_cluster), size=ntiles, p=probas)
        wsi = []
        infos = {} 
        composition = Counter(sel_clust)
        n = 0
        for c in composition.keys():
            pool = self.pools_c[c]
            ids = np.random.choice(np.arange(len(pool)),size=composition[c], replace=False)
            for x in pool[ids]:
                mat = np.load(self.get_path_from_ID(x[1]))
                wsi.append(mat[int(x[0]), :])
                infopath = self.get_path_from_ID(x[1], 'info')
                with open(infopath, 'rb') as f:
                    infodict = pickle.load(f)
                infos[n] = infodict[int(x[0])]
                infos[n]['name'] = x[1]
                n += 1
        return np.vstack(wsi), infos, composition

    def _sample_presence_single_pattern(self, cluster, classif, prop):
        probas = np.zeros(self.n_cluster)        
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
        probas = np.zeros(self.n_cluster)        
        for l in self.counter.keys():
            probas[l] = self.counter[l]
        if classif:
            probas[cluster] *= prop
        probas /= probas.sum()
        return probas

    def _sample_prop(self, up=0.5, down=1.5, cluster=None):
        if cluster is None:
            cluster = list(range(self.n_cluster))
        prop = np.ones(self.n_cluster)        
        prop[np.array(cluster)] = np.random.uniform(up, down, size=len(cluster))
        return prop
 
    def _sample_proportion_several_pattern(self, cluster, classif, prop):
        """
        changes the proportion of some clusters.
        If prop is None, then prop is set randomly.
        """
        if prop is None:
            if self.prop is None:
                self.prop = self._sample_prop(up, down, cluster)
        else:
            self.prop = np.ones(self.n_cluster)
            self.prop[np.array(cluster)] = np.array(prop)
        prop = self.prop
        probas = np.zeros(self.n_cluster)        
        for l in self.counter.keys():
            probas[l] = self.counter[l]
        if classif:
            probas *= prop
        probas /= probas.sum()
        return probas

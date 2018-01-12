"""
=========================================================================
                    Quantum Transport Clustering
=========================================================================
This package contains three major classes

    - GraphMethods: compute and store graph Laplacians
    - SpectralClustering: perform spectral clustering using graph Laplacians
    - QuantumTransportClustering: perform QTC using graph Laplacians

Code Author: Chenchao Zhao

The work was supervised by Professor Jun S. Song, supported by Sontag Foundation
and the Grainger Engineering Breakthroughs Initiative.

"""

import numpy as np
import scipy as sp
from scipy.sparse.csgraph import laplacian as sp_lap
from sklearn.cluster import KMeans
import math

pi = np.pi
palettes = [
    '#D870AD', '#B377D9', '#7277D5', 
    '#4B8CDC', '#3BB1D9', '#3BBEB0', 
    '#3BB85D', '#82C250', '#B0C151', 
    '#F5BA42', '#F59B43', '#E7663F', 
    '#D94C42', '#655D56', '#A2A2A2']
cc_ = np.array(palettes) # custom colors used in plots


class DataLoader:
    """
    
    Load data sets from csv file (no header) and store them as numpy arrays
    
    The rows are difference samples, while the columns are features, e.g. x, 
    y, z ...
    
    To load more complex data set, one may use other packages e.g. pandas.
    
    Initialization, e.g.
    
        >> data_ = DataLoader()
    
    - Use 'load' to load data, e.g. 
        
        >> data_.load("path/to/file", "nickname")
        
        The "nickname" is then used as the key for this data set. 
        If nickname is abscent, the filename or path string "path/to/file" is 
        used as the key for this data set.
    
    - Use 'get' to present data, e.g.
        
        >> X = data_.get("nickname")
    
        where X is a numpy array with rows as samples and columns as features
    
    - Use 'peek' to display loaded data sets, e.g.
    
        >> data_.peek()
        >> > data set nickname
    
    - Use 'delete' to remove a data set, e.g.
        
        >> data_.remove("nickname")
    
    - Use 'clear' to remove all loaded data sets, e.g.
    
        >> data_.clear()
    
    """
    
    def __init__(self, data_filename=None, nickname=None):
        self.datasets_ = dict()
        if data_filename != None:
            self.load(data_filename, nickname)
        
    
    def load(self, data_filename, nickname=None):
        """
        Load one data set from file
        """
        
        new_dataset = np.loadtxt(data_filename, delimiter=',')
        print(">> Data set {} loaded.".format(data_filename))
        
        if nickname != None:
            while nickname in self.datasets_:
                print(">>> nickname {} already taken.".format(nickname))
                nickname = input(">> new nickname: ")
            key = nickname
        else:
            key = data_filename
        
        self.datasets_[key] = new_dataset
        
    
    def get(self, key):
        """
        Return the specified data set as numpy array
        """
        return self.datasets_[key]
    
    def peek(self):
        """
        Print the list of data sets
        """
        for key in self.datasets_:
            print("> dataset: {}".format(key))
            
    def clear(self):
        """
        Delete all loaded data sets
        """
        self.datasets_ = dict()
        
    def delete(self, key):
        """
        Delete one data set
        """
        self.datasets_.pop(key, None)


class GraphMethods:
    """
    The Class GraphMethods is able to
    
        - Generate Gaussian RBF adjacency matrix using Euclidean distances of the 
          data distribution
        
        - Compute Graph Lapalcian (symmetrically normalized by default)
        
        - Store the raw data as well as adjacency matrix and graph Laplacian
    
    The Object is initialized with a data set, e.g.
        
        >> graph = GraphMethods(data_in_Euclidean_space)
        where the data is numpy array with rows as features, and columns as samples.
    
        If the data is an adjacency matrix of a undirected network, then one should 
        initialize it as follows:
        
        >> graph = GraphMethods(data_adjacency_matrix, graph_embedded = False)
        where graph_embedded is reset to False. The default is True.
    
    Other optional parameters:
    
        - int edt_tau: default None, the number of effective dissimilarity 
          transformation of Euclidean distance matrix. 
          Neglected if graph_embedded = False
    
        - float eps_quant: default 1, range (0, 100], the first eps quantile of 
          distance distribution among the data points. 
          Neglected if graph_embedded = False
    
        - bool normed: default True. If False, the graph Laplacian is unnormalized 
          L = D - A; if True, the graph Laplacian is symmetrically normalized by 
          sqrt of degree matrix, i.e. H = D^{-1/2} L D^{-1/2}
        - bool compute_lap: default is True. If True, graph Laplacian is computed after initialization.
    
    The Graph Laplacian can be obtained through attribute Lap_ if computed, e.g.
        
        >> graph.Lap_
    
    """
    
    def __init__(self, data_, graph_embedded=True, edt_tau = None, eps_quant = None, normed=True, compute_lap = True):
        
        self.graph_embedded = graph_embedded
        self.norm = normed
        
        if graph_embedded:
            
            self.Data_ = data_
            
            if edt_tau == None:
                self.edt_tau = 0 # default, no EDT iteration
            else:
                if edt_tau < 0 or not isinstance(edt_tau, int):
                    raise ValueError('edt_tau has to be a non-negative integer.')
                self.edt_tau = edt_tau
                
            if eps_quant == None:
                self.eps_quant = 1 # default 1 percent quantile
            else:
                if eps_quant > 0 and eps_quant <= 100:
                    self.eps_quant = eps_quant
                else:
                    raise ValueError('eps_quant has to be in the range (0,100].')
            
            print('> Initial parameters: graph is embedded in {} dim Euclidean space'.format(self.Data_.shape[0]))
            print('> EDT iterations: edt_tau = {}'.format(self.edt_tau))
            print('> Gaussian affinity eps quantile (%): eps_quant = {}'.format(self.eps_quant))
        
        else: 
            # not embedded in Euclidean space, then data is network adjacency
            if data_.shape[0] == data_.shape[1]:
                if np.any(data_ < 0):
                    raise ValueError('Entries of adjacency matrix should be non-negative.')
                self.Adj_ = (data_ + data_.T)/2
            else:
                raise ValueError('Adjacency matrix should be a symmetric matrix.')
                
            print('> Initial parameters: graph is not embedded in Euclidean space')
            
        if compute_lap:
            self.ComputeLaplacian()
    
    def distance_matrix_(self):
        """
        Data should be in the shape [n_features, n_samples]
        edt_tau is the number of EDT iterations
        """
        
        def euc_dist_mat_(X):
            temp = np.dot(X.T, X).round(15)
            d_ = np.diag(temp)
            d_1_ = np.outer(d_, np.ones_like(d_))
            d_ij_ = np.sqrt(d_1_ + d_1_.T - 2*temp)
            return d_ij_.round(15)
    
        def cos_dist_mat_(D):
            norm_ = D.sum(axis=0)
            X = np.sqrt(np.abs(D/norm_[:,None]))
            d_ij_ = 1-np.dot(X, X.T)
            return d_ij_.round(15)
        
        X_ = self.Data_
        D_ = euc_dist_mat_(X_)
        if self.edt_tau > 0:
            for t in range(int(self.edt_tau)):
                D_ = cos_dist_mat_(D_)
        self.Dist_ = D_
    
    def gaussian_rbf_adj_(self):        
        D_ = self.Dist_
        dist_ = np.sort(D_[D_>0])        
        r_eps = dist_[round(self.eps_quant * dist_.size / 100)]
        self.r_eps = r_eps
        self.Adj_ = np.exp(-D_**2 / r_eps**2)
    
    def adj_to_laplacian_(self):
        self.Lap_ = sp_lap(self.Adj_, normed=self.norm, return_diag=False)
        
    def ComputeLaplacian(self, output=False):
        if self.graph_embedded:
            self.distance_matrix_()
            print('>> Distance matrix done.')
            self.gaussian_rbf_adj_()
            print('>> Adjacency matrix done.')
        
        self.adj_to_laplacian_()
        print('>> Graph Laplacian done')
        
        if output:
            return self.Lap_
            
            
class SpectralClustering:
    """
    
    The class SpectralClustering is initialized by specifying the number of clusters: n_cluster
    
    Other optional parameters:
        - norm_method: None, "row" or "deg". Default is "row."
            If None, spectral embedding is not normalized;
            If "row," spectral embedding is normalized each row (each row represent a node);
            If "deg," spectral embedding is normalized by degree vector.
        - is_exact: bool. Default is True.
            If True, exact eigenvectors and eigenvalues will be computed.
            If False, first n_cluster low energy eigenvectors and eigenvalues (small eigenvalues) will be computed
    
    Method:
    
    fit (Laplacian_matrix) compute eigenvalue and eigenvectors of Laplacian_matrix and perform spectral embedding.
        
        >> clf = SpectralClustering(n_cluster=5)
        >> clf.fit(Laplacian)
    
    Attribute:
    
    labels_, a numpy array containing the class labels, e.g.
    
        >> clf.labels_
    
    Reference
    
    A Tutorial on Spectral Clustering, 2007 Ulrike von Luxburg http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

    
    """
    
    def __init__(self, n_clusters, norm_method='row', is_exact=True):
        self.nc = n_clusters
        self.is_exact = is_exact
        if norm_method == None:
            self.norm = 0 # do not normalize
        elif norm_method == 'row':
            self.norm = 1 # row normalize
        elif norm_method == 'deg':
            self.norm = 2 # deg normalize
        else:
            raise ValueError("norm_method can only be one of {None, 'row', 'deg'}.")
        
        print('> Initialization parameters: n_cluster={}'.format(self.nc))
        if self.norm == 0:
            print('> Unnormalized spectral embedding.')
        elif self.norm == 1:
            print('> Row normalized spectral embedding.')
        elif self.norm == 2:
            print('> Degree normalized spectral embedding.')

    def compute_eigs(self):
        if self.is_exact:
            self.Heigval, self.Heigvec = np.linalg.eigh(self.H_)
            print('> Exact eigs done')
        else:
            self.Heigval, self.Heigvec = sp.sparse.linalg.eigsh(self.H_, k=self.nc, which='SM')
            print('> Approximate eigs done')
    
    def fit(self, Lap_):
        
        self.H_ = Lap_
        self.compute_eigs()
        
        if self.norm == 0:
            spec_embed = self.Heigvec[:,:self.nc]
        elif self.norm == 1:
            spec_embed = (self.Heigvec[:,:self.nc].T/np.sqrt(np.sum(self.Heigvec[:,:self.nc]**2, axis=1))).T
        elif self.norm == 2:
            spec_embed = (self.Heigvec[:,:self.nc].T / np.abs(self.Heigvec[:,0])).T
        
        km_ = KMeans(n_clusters = self.nc, n_init=100).fit(spec_embed)
        self.labels_ = km_.labels_        
    

class QuantumTransportClustering:
    """
    
    The Class QuantumTransportClustering is initialized by specifying:
        - n_cluster: the number clusters
        - Hamiltonian: a symmetric graph Laplacian
    
    Other optional paramters:
        - s = 1.0: used to generate the Laplace transform s-parameter, 
          which is s * (E[n_cluster - 1]/(n_cluster - 1)).
        - is_exact = True: if True, exact eigenvalue and eigenvectors 
          of graph Laplacian will be computed
        - n_eigs = None: If is_exact = False, first n_eigs low energy 
          eigenvectors and eigenvalues will be computed, by default 
          n_eigs = min(10*n_clusters, number_of_nodes). If is_exact 
          = True, but n_eigs is not None, then first n_eigs low energy
          exact eigenstates will be used in QT clustering.
    
    Methods:
    
        - Grind():
          Generate quantum transport with a set of initialization nodes and extract their phase distributions.
          The phase distributions are mapped to a set of class label vectors ranging from 0 to n_cluster-1
          The class label vectors are stored in matrix "Omega_"
    
        - Espresso()
          Apply "direct extraction method" to raw labels Omega_
          The final decision can be obtained using "labels_"
    
        - Coldbrew()
          Apply "consensus matrix method" to raw labels Omega_
          The consensus matrix can be obtained using "consensus_matrix_"
    
    Attributes:
    
        - Omega_, the Omega matrix whose columns are class labels associated with initialization nodes
    
        - labels_, the predicted class labels using direct extraction or "Espresso()" method
    
        - consensus_matrix_, the consensus matrix based on Omega matrix using "Coldbrew()" method
    
    """
    
    machine_eps = np.finfo(float).eps
    
    def __init__(self, n_clusters, Hamiltonian, s=1.0, is_exact=True, n_eigs = None):
        
        if Hamiltonian.shape[0] == Hamiltonian.shape[1]:
            self.m_nodes = Hamiltonian.shape[0]
            n_primes = self.m_nodes
            self.H_ = Hamiltonian
        else:
            raise ValueError('Hamiltonian or Graph Laplacian should be a symmetric matrix.')
    
    
        if isinstance(n_clusters, int) and n_clusters > 1:
            self.nc = n_clusters
            print('> Initialization parameters: n_cluster={}'.format(self.nc))
        else:
            raise ValueError('Number of clusters is an int and n_clusters > 1.')
    
        self.is_exact = is_exact
        if not is_exact:
            self._n_eigs = min(10*n_clusters, self.m_nodes) if n_eigs == None else n_eigs
            print('> First {} low energy eigenvalues will be computed.'.format(self._n_eigs))
    
        self.n_eigs = n_eigs
    
        self.s = s
        print('> Laplace variable s = {}'.format(self.s))
    
        self.primes_ = self.generate_primes(n_primes)
    
        print('> First {} primes generated'.format(n_primes))     
        print('>> Espresso: direct extraction method')
        print('>> Coldbrew: consensus matrix method')
            
        

    def compute_eigs(self):
        if self.is_exact:
            self.Heigval, self.Heigvec = np.linalg.eigh(self.H_)
            print('> Exact eigs done')
        else:
            self.Heigval, self.Heigvec = sp.sparse.linalg.eigsh(self.H_, k=self._n_eigs, which='SM')
            print('> Approximate eigs done')
        
        gaps_ = np.abs(np.diff(self.Heigval))
        if np.any(gaps_ < self.machine_eps):
            print('> Warning: Some energy gaps are smaller than machine epsilon. QTC results may show numerial instability.')
    
    def generate_primes(self, n):
        """
        Find first n primes
        """
        if n <= 0:
            raise ValueError('n_prime is int and > 0.')
        elif n == 1:
            return [1]
        elif n == 2:
            return [1,2]
        elif n == 3:
            return [1,2,3]
        else:
            primes_ = [1,2,3]
            itr = n - 3

        new = primes_[-1]

        while itr > 0:
            new += 1
            is_prime = False

            if new & 1 == 1:
                is_prime = True
                for k in range(3, int(math.floor(math.sqrt(new)))+1):
                    if new % k == 0:
                        is_prime = False
                        break
            if is_prime:
                primes_.append(new)
                itr -= 1
        return np.array(primes_)
    
    def laplace_transform_wf_(self, s, eigval_, eigvec_, init_vec_):
        # expand init_vec_ in terms of eig_vec_
        coeff_ = np.dot(init_vec_, eigvec_)
        w_ = coeff_/(s+1j*eigval_)
        psi_s_ = np.dot(eigvec_, w_)
        return psi_s_
    
    def phase_info_clustering_diff_(self, phase_, n_cluster=None):
        if n_cluster == None:
            n_cluster = self.nc
        elif n_cluster == 1:
            class_label_ = np.zeros(phase_.size)
        else:
            while np.any(phase_ < 0):
                phase_[phase_<0] += 2*pi
            while np.any(phase_ > 2*pi):
                phase_[phase_>2*pi] -= 2*pi   
            idx_ = np.argsort(phase_)
            iidx_ = np.argsort(idx_)
            z_ = np.exp(1j*phase_[idx_])
            diff_ = np.zeros(z_.size, dtype=float)
            diff_[0] = np.abs(z_[0] - z_[-1])
            diff_[1:] = np.abs(np.diff(z_))
            n_part = n_cluster
            partition_idx_ = np.argpartition(diff_, -n_part)[-n_part:]
            partition_idx_ = np.sort(partition_idx_)
            class_label_ = np.zeros_like(idx_)
            for k in range(1,n_cluster):
                class_label_[partition_idx_[k-1]:partition_idx_[k]] += k
            class_label_ = class_label_[iidx_]
        return class_label_
    
    def phase_info_clustering_KMeans_(self, phase_, n_cluster):
        if n_cluster == None:
            n_cluster = self.nc
        elif n_cluster == 1:
            class_label_ = np.zeros(phase_.size)
        else:
            z_ = np.exp(1j*phase_)
            data_ = np.vstack((z_.real, z_.imag)).T
            km = KMeans(n_clusters=n_cluster)
            km.fit(data_)
            class_label_ = km.labels_
        return class_label_
    
    def Grind(self, s=None, grind='medium', method='diff', init_nodes_=None):
        """
        grind option can be "coarse", "medium", "fine", "micro", "custom"
        If grind="custom" one need to specify the a list of nodes as init_nodes_
        """
        
        if s == None:
            s = self.s
        else:
            self.s = s
            print('> Update: Laplace variable s = {}'.format(s))
        
        
        if grind == 'coarse':
            _every_ = self.m_nodes // 30
        elif grind == 'medium':
            _every_ = self.m_nodes // 60
        elif grind == 'fine':
            _every_ = self.m_nodes // 100
        elif grind == 'micro':
            _every_ = 1
        elif grind == 'custom':
            if init_nodes_ == None:
                raise ValueError('> If grind is custom, you need to specify a list of initialization nodes, e.g. init_nodes_=[0,1,2,3,10]')
            else:
                init_nodes_ = np.array(init_nodes_)
                _every_ = None
        else:
            raise ValueError('> parameter grind can be {coarse, medium, fine, micro, or custom}.')
        
        self.compute_eigs() # compute Heigval and Heigvec
        
        deg_idx_ = np.argsort(self.Heigvec[:,0]**2)
        
        if _every_ != None:
            if _every_ == 0: _every_ = 1
            init_nodes_ = deg_idx_[::_every_]
        
        m_init = init_nodes_.size
        self.m_init = m_init
        print('> {}-ground: {} initialization nodes'.format(grind, m_init))
        
        Omega_ = np.zeros((self.m_nodes, m_init), dtype=int) # col for init_
        
        show_warning = False
        
        for jdx, idx in enumerate(init_nodes_):
            init_ = np.zeros(self.m_nodes)
            init_[idx] += 1.0
            psi_s_ = self.laplace_transform_wf_(s*(self.Heigval[self.nc-1]-self.Heigval[0])/(self.nc-1), \
            self.Heigval[:self.n_eigs]-self.Heigval[0], self.Heigvec[:,:self.n_eigs], init_)
            rho_s_ = np.abs(psi_s_)
            theta_s_ = np.angle(psi_s_)
            
            if np.any(rho_s_ < self.machine_eps):
                show_warning = True
#                 theta_s_[np.where(psi_s_ < self.machine_eps)] = 0.0
#                 print('! Small amplitude warning !')
            
            if method == 'diff':
                Omega_[:,jdx] = self.phase_info_clustering_diff_(theta_s_, self.nc)
            elif method == 'kmeans':
                Omega_[:,jdx] = self.phase_info_clustering_KMeans_(theta_s_, self.nc)
            else:
                raise ValueError('> method can only be {diff, or kmeans}.')

        self.Omega_ = Omega_
        if show_warning:
            print('>> Warning: some amplitudes are below machine eps. QTC may show numerical instability.')
        return Omega_
    
    def is_equiv(self, Omg_cols_):
    
        p_ = np.sqrt(self.primes_) 

        unique_col_ = np.unique(Omg_cols_, axis=1, return_counts=False)
        n_cluster = self.nc

        mix_all_ = unique_col_ @ p_[:unique_col_.shape[1]]
        if np.unique(mix_all_).size == n_cluster:
            return True
        else:
            return False
    
    def pigeonhole(self, columns_, columns_idx=None, hashtab=None):
    
        if columns_idx == None:
            columns_idx = list(range(columns_.shape[1]))

        n_cols_ = len(columns_idx)

        if hashtab == None:
            hashtab = dict()


        if self.is_equiv(columns_[:,columns_idx]):
            is_existing = False
            for hashtag in hashtab.keys():
                if self.is_equiv(columns_[:,[columns_idx[0],hashtag]]):
                    hashtab[hashtag] += columns_idx
                    is_existing = True

            if not is_existing:
                new_hashtag = columns_idx[0]            
                hashtab[new_hashtag] = columns_idx
        else:
            columns_idx_1 = columns_idx[:n_cols_//2]
            columns_idx_2 = columns_idx[n_cols_//2:]        

            hashtab = self.pigeonhole(columns_, columns_idx_1, hashtab)
            hashtab = self.pigeonhole(columns_, columns_idx_2, hashtab)

        return hashtab
        
    
    def Espresso(self):
        Table_ = self.pigeonhole(self.Omega_)
        
        alpha_ = np.array(list(Table_.keys()))
        weight_alpha_ = np.zeros(alpha_.size)
        
        for j in range(alpha_.size):
            weight_alpha_[j] = len(Table_[alpha_[j]])
            
        sort_idx_ = np.argsort(weight_alpha_)[::-1]
        weight_alpha_ = weight_alpha_[sort_idx_]
        weight_alpha_ /=  weight_alpha_.sum()
        alpha_ = alpha_[sort_idx_]
        
        self.double_shot_ = (alpha_, weight_alpha_)
        self.labels_ = self.Omega_[:,alpha_[0]]
        
    def Coldbrew(self):
        C_ = np.diag([self.m_init]*self.m_nodes)
        for idx in range(self.m_nodes):
            for jdx in range(idx+1, self.m_nodes):
                temp = np.sum(self.Omega_[idx,:] == self.Omega_[jdx, :])
                C_[idx, jdx] = temp
                C_[jdx, idx] = temp
                
        self.consensus_matrix_ = C_

def main():
    print(">> Type 'exit' to quit")
    print(">> Type 'load' to load date set") 
    print(">> Type 'show' to display loaded data sets")
    print(">> Type 'demo' to run demonstation")
    datasets_ = DataLoader()
    graphs_ = dict()
    spectral_clf = dict()
    qtc_clf = dict()
    
    while True:
        command = input("new command >> ").strip()
        
        if command == 'exit':
            break
        elif command == 'load':          
            filename = input("filename or path to file: ").strip()
            nickname = input("nickname for this data set: ").strip()
            datasets_.load(filename, nickname)
            print(">> Available data sets: ")
            datasets_.peek()
            print(">> Type 'clear' to remove all loaded date sets")
            
        elif command == 'clear':
            datasets_.clear()
            
        elif command == 'graph':         
            dataset_key = input("Compute graph Laplacian of data set ")
            eps = float(input("The short proximity length scale is chosen from quantile (%): "))
            
            
            data_ = datasets_.get(dataset_key).T
            g_ = GraphMethods(data_, eps_quant = eps)
            graphs_[dataset_key] = g_
            print("Use command 'spectral' for spectral clustering ...")
            print("Use command 'qtc' for the QT clustering ...")
            
        elif command == 'spectral':
            dataset_key = input("Apply spectral clustering to data set ")
            if dataset_key in graphs_:
                n_cluster = int(input("Please specify the number of clusters "))
                clf = SpectralClustering(n_cluster, 'row')
                clf.fit(graphs_[dataset_key].Lap_)
                spectral_clf[dataset_key] = clf
            else:
                print(">> Please compute graph Laplacian first using the command 'graph'.")
        elif command == 'qtc':
            dataset_key = input("Apply quantum transport clustering to data set ")
            if dataset_key in graphs_:
                n_cluster = int(input("Please specify the number of clusters ").strip())
                lap_ = graphs_[dataset_key].Lap_
                qtc = QuantumTransportClustering(n_cluster, lap_, s=1, is_exact=False, n_eigs=min(10*n_cluster, lap_.shape[0]))
                Omega_ = qtc.Grind(grind='fine', method='diff')
                qtc.Espresso()
                qtc_clf[dataset_key] = qtc
            else:
                print(">> Please compute graph Laplacian first using the command 'graph'.")
        elif command == 'show':
            print(">>> Data sets:")
            datasets_.peek()
            print(">>> Graph Objects:")
            for key in graphs_:
                print("> graph:  {}".format(key))
                
            print(">>> Spectral clustering applied to ...")
            for key in spectral_clf:
                print("> spec:   {}".format(key))
            print(">>> QT clustering applied to ...")
            for key in qtc_clf:
                print("> qtc:    {}".format(key))
        elif command == 'plot spectral':
            dataset_key = input("Plot spectral clustering of data set ")
            if dataset_key in spectral_clf:
                data_ = datasets_.get(dataset_key).T
                ll_ = spectral_clf[dataset_key].labels_
                if ll_.max() > cc_.size:
                    c_ = ll_
                elif ll_.max() > cc_.size // 2:
                    c_ = cc_[ll_]
                else:
                    c_ = cc_[ll_*2]
                plt.scatter(data_[0,:], data_[1,:], s=30, c='', edgecolors=c_, alpha=0.6, linewidth=0.8)
                plt.show()
            else:
                print(">> Please perform spectral clustering first")
        elif command == 'plot qtc':
            dataset_key = input("Plot QT clustering of data set ")
            if dataset_key in qtc_clf:
                data_ = datasets_.get(dataset_key).T
                ll_ = qtc_clf[dataset_key].labels_
                if ll_.max() > cc_.size:
                    c_ = ll_
                elif ll_.max() > cc_.size // 2:
                    c_ = cc_[ll_]
                else:
                    c_ = cc_[ll_*2]
                plt.scatter(data_[0,:], data_[1,:], s=30, c='', edgecolors=c_, alpha=0.6, linewidth=0.8)
                plt.show()
            else:
                print(">> Please perform QT clustering first")
        elif command == 'demo':
            data_ = DataLoader()
            data_.load("./data/stick", "easy") # path to file, nickname
            data_.load("./data/thunder", "thor")
            data_.load("./data/annuli", "ring")
            
            graphs_ = dict.fromkeys(["easy", "thor", "ring"], None)
            eps_q_ = {"easy": 3.5, "thor": 0.8, "ring": 0.4} # the epsilon quantile of distance distribution
            for key in graphs_:
                print("[data set {}]".format(key))
                graphs_[key] = GraphMethods(data_.get(key).T, eps_quant=eps_q_[key])
                
            print(">> Stick Data Set:")
            spec = SpectralClustering(n_clusters=3, norm_method='row')
            spec.fit(graphs_["easy"].Lap_)
            shot = QuantumTransportClustering(n_clusters=3, Hamiltonian=graphs_["easy"].Lap_)
            shot.Grind() # generate quantum transport from 60 initialization nodes
            shot.Espresso() # apply direct extract method
            
            X = data_.get("easy").T
            fig = plt.figure(figsize=(20, 15))
            ax = fig.add_subplot(321, aspect="equal")
            ax.scatter(X[0,:], X[1,:], s=20, c='', edgecolors=cc_[spec.labels_*2])
            plt.xlim(-1,1)
            plt.ylim(-1,1)
            plt.title("Spectral Clustering")
            ax = fig.add_subplot(322, aspect="equal")
            ax.scatter(X[0,:], X[1,:], s=20, c='', edgecolors=cc_[shot.labels_*2])
            plt.xlim(-1,1)
            plt.ylim(-1,1)
            plt.title("QT Clustering")

            
            
            print(">> Thunder Data Set:")
            spec = SpectralClustering(n_clusters=8, norm_method='row')
            spec.fit(graphs_["thor"].Lap_)            
            shot = QuantumTransportClustering(n_clusters=8, Hamiltonian=graphs_["thor"].Lap_, n_eigs=100)
            shot.Grind(grind="fine") # increase the number of initialization nodes to improve clustering accuracy
            shot.Espresso() # apply direct extraction
            
            X = data_.get("thor").T
            ax = fig.add_subplot(323, aspect="equal")
            ax.scatter(X[0,:], X[1,:], s=20, c='', edgecolors=cc_[spec.labels_*2])
            plt.xlim(-1,1)
            plt.ylim(-1,1)
            plt.title("Spectral Clustering")
            ax = fig.add_subplot(324, aspect="equal")
            ax.scatter(X[0,:], X[1,:], s=20, c='', edgecolors=cc_[shot.labels_*2])
            plt.xlim(-1,1)
            plt.ylim(-1,1)
            plt.title("QT Clustering")

            
            print(">> Annuli Data Set:")
            spec = SpectralClustering(n_clusters=5, norm_method='row')
            spec.fit(graphs_["ring"].Lap_)
            shot = QuantumTransportClustering(n_clusters=5, Hamiltonian=graphs_["ring"].Lap_, n_eigs=100)
            X = data_.get("ring").T
            idx_ = np.argsort(X[0,:]**2 + X[1,:]**2).tolist() # sort the data points according to r^2 = x^2 + y^2
            shot.Grind(grind="custom", init_nodes_=idx_[:200:2]) # use points close to origin as initialization nodes to resolve the fine structures at the center
            shot.Espresso()
            
            ax = fig.add_subplot(325, aspect="equal")
            ax.scatter(X[0,:], X[1,:], s=20, c='', edgecolors=cc_[spec.labels_*3])
            plt.xlim(-1,1)
            plt.ylim(-1,1)
            plt.title("Spectral Clustering")
            ax = fig.add_subplot(326, aspect="equal")
            ax.scatter(X[0,:], X[1,:], s=20, c='', edgecolors=cc_[shot.labels_*3])
            plt.xlim(-1,1)
            plt.ylim(-1,1)
            plt.title("QT Clustering")
            plt.show()
            
            data_.clear()        
        else:
            print(">> Unknown command")
            

            
        
        
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    main()

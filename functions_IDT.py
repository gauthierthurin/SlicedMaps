from scipy.stats import ortho_group
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

def random_orthonormal_matrix(n,seed=None):
    return ortho_group.rvs(dim=n,random_state=seed)
 
def transport_map_1d(x_source, y_target):
    """
    Returns the optimal transport map T such that T(X) ~ Y,
    and the image T(x_source).

    Parameters
    ----------
    x_source : Source sample (1D)
    y_target : Target sample (1D)
    """
    
    x_source = np.asarray(x_source, dtype=float)
    y_target = np.asarray(y_target, dtype=float)

    # Sort both samples
    xs_sorted = np.sort(x_source)
    ys_sorted = np.sort(y_target)

    # Compute ranks of each source point
    ranks = np.argsort(np.argsort(x_source))  # Gives a value in {0,...,n-1}

    # In 1D, the optimal transport map is the quantile-to-quantile matching
    Tx = ys_sorted[ranks]

    # Also return the function T(x), usable on any new query point
    def T(x):
        """
        Optimal transport map computed via empirical CDF interpolation.
        """
        x = np.asarray(x)
        # Compute empirical CDF values of x relative to the source distribution
        cdf = np.searchsorted(xs_sorted, x, side="right") / len(xs_sorted)
        # Map through the target quantile function
        idx = np.clip((cdf * len(ys_sorted)).astype(int), 0, len(ys_sorted)-1)
        return ys_sorted[idx]

    return T, Tx 

def transport_map_1d_rowwise(X_source, Y_target):
    """
    Computes the optimal transport image Tx for each pair of rows
    in X_source and Y_target. Each row is treated as a separate 1D sample.

    Parameters
    ----------
    X_source : array-like, shape (N, n1)
        Matrix where each row is a source 1D sample.
    Y_target : array-like, shape (N, n2)
        Matrix where each row is a target 1D sample.

    Returns
    -------
    Tx_matrix : np.ndarray, shape (N, n1)
        Matrix where each row contains the optimal transport image of the
        corresponding row in X_source.
    """

    X_source = np.asarray(X_source, dtype=float)
    Y_target = np.asarray(Y_target, dtype=float)

    if X_source.shape[0] != Y_target.shape[0]:
        raise ValueError("X_source and Y_target must have the same number of rows.")

    N, n1 = X_source.shape
    _, n2 = Y_target.shape

    # 1) Compute ranks of X row-wise (argsort twice to get rank of each element)
    rank1 = np.argsort(X_source, axis=1)
    ranks = np.argsort(rank1, axis=1)  # shape (N, n1)

    # 2) Sort Y row-wise 
    Y_sorted = np.take_along_axis(Y_target, np.argsort(Y_target, axis=1), axis=1) # shape: (N, n2)

    # 3) OT map: Tx = sorted_Y[row][ rank_of_each_x ]
    # We want: Tx[i, j] = Y_sorted[i, ranks[i, j]]
    Tx_matrix = np.take_along_axis(Y_sorted, ranks, axis=1)

    return Tx_matrix 

def DistributionTransfer(X_source,Y_target,P):
    """ 
    Computes one step of the Iterative Distribution Transfer algorithm. 
    Equivalently, it computes a slice-matching map T along the orthonormal basis P. 

    Parameters
    ----------
    X_source : array-like, shape (N, n1)
        Matrix where each row is a source 1D sample.
    Y_target : array-like, shape (N, n2)
        Matrix where each row is a target 1D sample.

    Returns
    -------
    Tx_matrix : np.ndarray, shape (N, n1)
        Matrix of images T(x) for each point from X_source
    """
    X_s_projections = np.dot(X_source, P)
    X_t_projections = np.dot(Y_target, P)
    Tx_matrix = transport_map_1d_rowwise(X_s_projections, X_t_projections)
    Tx_matrix = np.dot(Tx_matrix,P.T)
    return(Tx_matrix)

def IDT(X_source,Y_target,K = 5,alpha=0.51,seed=None):
    """ 
    Computes K steps of the Iterative Distribution Transfer algorithm. 

    Parameters
    ----------
    X_source : array-like, shape (N, n1)
        Matrix where each row is a source 1D sample.
    Y_target : array-like, shape (N, n2)
        Matrix where each row is a target 1D sample.
    K : Number of steps for IDT.
    alpha: Power of the learning rate.

    Returns
    -------
    Tx_matrix : np.ndarray, shape (N, n1)
        Matrix of images T(x) for each point from X_source 
    """
    dim = X_source.shape[1]
    t = (1/np.linspace(1,K,K))**alpha
    T_x0 = X_source # initialization
    if seed==None:
        seed = np.random.randint(0,1000)
    xt = []
    for i in range(K):
        P = random_orthonormal_matrix(dim,seed*i)
        T_x0 = (1-t[i])*T_x0+ t[i]*DistributionTransfer(T_x0,Y_target,P)
        xt.append(T_x0) 
    xt = np.array(xt)
    return( xt )

from ot.sliced import sliced_wasserstein_distance 
def compute_loss_variousdata(n=500,d=2, K=10,Nrep=20,alpha=0.51,target=None):
    """ 
    Computes IDT and the loss at each iteration. 
    Results are repeated several times, with new simulated datasets and random orthonormal basis along iterations. 

    The source distribution is a discrete sample from a mixture of isotropic Gaussians. 

    The target distribution is a discrete sample from different mixture of isotropic Gaussians if target = None. 
    If target is "StandardGaussian" or "CovGaussian", then it comes respectively from a isotropic Gaussian and from a Gaussian with non-isotropic covariance.  

    ----------
    Parameters:
    ----------
    n: number of samples for each simulated dataset.
    d: dimension of samples for each simulated dataset.
    K: number of steps for the IDT algorithm
    Nrep : amount of repetitions. 
    alpha: choice of stepsize in IDT
    """

    list_res = [] 
    for rep in range(Nrep):
        # Source dataset 
        x0 = sample_blobs(n=n,dim=d,centers=4,random_state=None) + 10
        # Target dataset
        x1 = sample_blobs(n=n,dim=d,centers=3,random_state=None) 
        if target=="StandardGaussian": 
            x0 = sample_blobs(n=n,dim=d,centers=4,random_state=None)/3
            x1 = np.random.multivariate_normal(np.zeros(d),np.eye(d),size=n)
        if target=="CovGaussian":
            x0 = sample_blobs(n=n,dim=d,centers=4,random_state=None)/3
            A = random_covariance_mat(d)
            x1 = np.random.multivariate_normal(np.zeros(d),A,size=n) 
        xt = IDT(x0,x1,K=K,alpha=alpha)
        SW_ = []
        SW_.append(sliced_wasserstein_distance(x0,x1))
        for k in range(xt.shape[0]): 
            SW_.append(sliced_wasserstein_distance(xt[k] ,x1))
        list_res.append(SW_)
    list_res = np.array(list_res) 
    median = np.median(list_res, axis=0)
    q1 = np.percentile(list_res, 5, axis=0)
    q3 = np.percentile(list_res, 95, axis=0)
    return(median,q1,q3)




################################################################################################
########## GENERATE DATASETS, VIZUALISATIONS 
################################################################################################
import torch
from torchdyn.datasets import generate_moons

def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), np.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data

def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.2)
    return np.array(x0 * 3 - 1)

def sample_8gaussians(n):
    return np.array(eight_normal_sample(n, 2, scale=5, var=0.1).float())

from sklearn.datasets import make_blobs
def sample_blobs(n,dim,centers=4,random_state=None):
    X, _ = make_blobs(n_samples=n, centers=centers, n_features=dim,random_state=random_state)
    return(X)

def random_covariance_mat(d, min_eig=0.1, max_eig=2.0, seed=None):
    rng = np.random.default_rng(seed)

    # random orthogonal matrix
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))

    # random eigenvalues
    eigvals = rng.uniform(min_eig, max_eig, size=d)

    cov = Q @ np.diag(eigvals) @ Q.T
    return cov

################################################################################################
########## FUNCTIONS FOR CONTINUOUS IDT : EXPLICIT UPDATES WITH COVARIANCES MATRICES OF GAUSSIANS:
################################################################################################

def IDT_linear(cov0,cov1,K = 10,alpha=0.1,seed=62):
    """ 
    Runs and evaluates the IDT algorithm under linear structure: if both x0 and x1 are Gaussian distributions. 

    Parameters
    ----------
    X_source : array-like, shape (N, n1)
        Matrix where each row is a source 1D sample.
    Y_target : array-like, shape (N, n2)
        Matrix where each row is a target 1D sample.
    K : Number of steps for IDT.
    alpha: Power of the learning rate.
    seed: random seed for orthonormal matrices. 

    Returns
    -------
    list_cov : list of covariance matrices along the iterates. 
    list_eig_min, list_eig_max: minimum and maximum eigenvalue along the iterates. 
    list_SW: list of SW-distance between the k-th iterate and the target. 
    """
    dim = cov0.shape[1]
    t = 1 /np.linspace(1,K,K)**alpha
    list_eig_min = []
    list_eig_max = []
    list_SW = []
    list_cov = []
    cov_k = cov0
    if seed == None:
        seed = np.random.randint(0,1000)
    for i in range(K):
        P = random_orthonormal_matrix(dim,seed*i)
        var0 = np.sum(P * (cov_k @ P) , axis=0) # theta^T cov0 theta
        var1 = np.sum(P * (cov1 @ P) , axis=0) # theta^T cov1 theta
        PDP = P @ np.diag( np.sqrt( var1 / var0 ) ) @ P.T 

        eigenvalues, eigenvectors = np.linalg.eig(cov_k) 
        list_eig_min.append(np.min(eigenvalues))
        list_eig_max.append(np.max(eigenvalues))

        SWk = np.linalg.norm(np.sqrt( var1) - np.sqrt(var0) )
        list_SW.append( SWk )  

        interpol = (1-t[i])*np.eye(dim)+ t[i]*PDP 
        cov_k = interpol @ cov_k @ interpol
        list_cov.append(cov_k) 
    return( np.array(list_cov) , np.array(list_eig_min),np.array(list_eig_max), np.array(list_SW) )


def IDT_linear_singleTheta(cov0,cov1,K = 10,alpha=0.1,seed=62):
    """ 
    Runs and evaluates the IDT algorithm under linear structure: if both x0 and x1 are Gaussian distributions. 
    It uses a single direction theta at each iteration, instead of a whole orthonormal basis of directions. 
    """
    dim = cov0.shape[1]
    t = 1 /np.linspace(1,K,K)**alpha
    list_eig_min = []
    list_eig_max = []
    list_SW = []
    list_cov = []
    cov_k = cov0
    if seed == None:
        seed = np.random.randint(0,1000)
    for i in range(K):
        P = random_orthonormal_matrix(dim,seed*i)
        var0 = np.sum(P * (cov_k @ P) , axis=0) # theta^T cov0 theta
        var1 = np.sum(P * (cov1 @ P) , axis=0) # theta^T cov1 theta
        diag = np.zeros(dim)
        diag[0] = 1 
        PDP = P @ np.diag( diag* np.sqrt( var1 / var0 ) + np.ones(dim) -  diag ) @ P.T 

        eigenvalues, eigenvectors = np.linalg.eig(cov_k) 
        list_eig_min.append(np.min(eigenvalues))
        list_eig_max.append(np.max(eigenvalues))

        SWk = np.linalg.norm(np.sqrt( var1) - np.sqrt(var0) )
        list_SW.append( SWk )  

        interpol = (1-t[i])*np.eye(dim)+ t[i]*PDP 
        cov_k = interpol @ cov_k @ interpol
        list_cov.append(cov_k) 
    return( np.array(list_cov) , np.array(list_eig_min),np.array(list_eig_max), np.array(list_SW) )


def rep_IDT(Nrep,dim,K,alpha,seed,cov1):
    list_rep_SW,list_rep_eig_min,list_rep_eig_max = [],[],[]
    for rep in range(Nrep):
        cov0 = random_covariance_mat(dim)
        list_cov, list_eig_min, list_eig_max, list_SW =  IDT_linear(cov0,cov1,K = K,alpha=alpha,seed=seed)
        list_rep_SW.append(list_SW)
        list_rep_eig_min.append(list_eig_min)
        list_rep_eig_max.append(list_eig_max)
    return(np.array(list_rep_SW), list_rep_eig_min, list_rep_eig_max) 


def rep_IDT_singleTheta(Nrep,dim,K,alpha,seed,cov1):
    list_rep_SW,list_rep_eig_min,list_rep_eig_max = [],[],[]
    for rep in range(Nrep):
        cov0 = random_covariance_mat(dim)
        list_cov, list_eig_min, list_eig_max, list_SW =  IDT_linear_singleTheta(cov0,cov1,K = K,alpha=alpha,seed=seed)
        list_rep_SW.append(list_SW)
        list_rep_eig_min.append(list_eig_min)
        list_rep_eig_max.append(list_eig_max)
    return(np.array(list_rep_SW), list_rep_eig_min, list_rep_eig_max) 


def rep_IDT_BothAlgos(Nrep,dim,K,alpha,seed,cov1):
    list_rep_eig_min,list_rep_eig_max,list_rep_eig_min_tht, list_rep_eig_max_tht = [],[],[],[]
    for rep in range(Nrep):
        cov0 = random_covariance_mat(dim)
        list_cov, list_eig_min, list_eig_max, list_SW =  IDT_linear(cov0,cov1,K = K,alpha=alpha,seed=seed)
        list_cov_tht, list_eig_min_tht, list_eig_max_tht, list_SW_tht =  IDT_linear_singleTheta(cov0,cov1,K = K,alpha=alpha,seed=seed)
        list_rep_eig_min.append(list_eig_min)
        list_rep_eig_max.append(list_eig_max)
        list_rep_eig_min_tht.append(list_eig_min_tht)
        list_rep_eig_max_tht.append(list_eig_max_tht)
    return(list_rep_eig_min, list_rep_eig_max, list_rep_eig_min_tht, list_rep_eig_max_tht) 

def plot_SW_IDT(list_medians,list_q1,list_q3,list_dim,alpha,NameDataset):
    k_values = np.arange(list_medians[0].shape[0])  
    colors = sns.color_palette("flare") 

    plt.figure(figsize=(3,3))
    plt.style.use('seaborn-v0_8')
    c = 0 
    for dim in list_dim:
        plt.plot(k_values, list_medians[c], label='{}'.format(dim), color=colors[c])
        plt.fill_between(k_values, list_q1[c], list_q3[c], alpha=0.3, color=colors[c])
        c += 1 
    plt.legend(loc = "upper right",ncol=2,frameon=True,fontsize=12)
    plt.grid(True) 
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("FiguresIDT/curve_loss_{}_alpha{}.pdf".format(NameDataset,alpha),bbox_inches='tight')


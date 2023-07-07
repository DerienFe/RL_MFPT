import numpy as np
import torch
import torch.linalg as linalg
from numpy.linalg import inv
KbT = 0.596
"""
A catch-all file for all code relating to kinetic networks
"""


def compute_free_energy(K, KbT):
    beta = np.divide(1, KbT)
    eigenvalues, eigenvectors = linalg.eig(K)
    eigenvalues_sorted = np.sort(eigenvalues.real)[::-1]
    indices = np.argsort(eigenvalues.real)[::-1]
    largest_ev = eigenvectors[:, indices[0]]
    pi = largest_ev.T / sum(largest_ev)
    free_energy = np.divide(-np.log(pi), beta)
    return (pi,
            free_energy,
            eigenvectors,
            eigenvalues,
            eigenvalues_sorted,
            indices)


def Markov_mfpt_calc(peq, M, N):
    onevec = np.ones((N, 1))
    Idn = np.diag(onevec[:, 0])
    A = peq.T @ onevec.T
    A = A.T
    Qinv = np.linalg.inv(Idn + A - M)
    mfpt = np.zeros((N, N))
    for j in range(N):
        for i in range(N):
            term1 = Qinv[j, j] - Qinv[i, j] + Idn[i, j]
            mfpt[i, j] = 1/peq[0, j] * term1
    return mfpt


def mfpt_peq_calc(transistion_matrix):
    if isinstance(transistion_matrix, str):
        markov = torch.tensor(np.loadtxt(transistion_matrix)).double()
    elif not isinstance(transistion_matrix, torch.Tensor):
        transistion_matrix = torch.tensor(transistion_matrix)
    eigenvalues, eigenvectors = linalg.eig(markov.t())
    eigenvalues_sorted, indices = torch.sort(eigenvalues.real, descending=True)
    peq = eigenvectors[:, indices[0]].real
    peq = torch.unsqueeze(peq, 1)
    mfpt = compute_mfpt_msm(peq, markov)
    return mfpt, peq


def compute_mfpt_msm(peq, msm):
    """
    Calculates the mean first passage times of the transition probability
    matrix.

    Parameters
    ----------
    peq : torch.Tensor(n_nodes)
        Eigenvector of the stationary distribution.
    msm : torch.Tensor(n_nodes,n_nodes)
        Transition probability matrix.

    Returns
    -------
    mfpt : torch.Tensor(n_nodes,n_nodes)
        Mean first passage time matrix.
    """
    n_nodes = peq.shape[0]
    identity = torch.eye(n_nodes, dtype=torch.float64)
    onevec = torch.ones((1, n_nodes), dtype=torch.float64)
    peq = peq.reshape(peq.shape[0], 1)

    q_inv = torch.linalg.inv(identity+torch.matmul(peq, onevec).t()-msm)

    mfpt = torch.zeros((n_nodes, n_nodes), dtype=torch.float64)
    for j in range(n_nodes):
        for i in range(n_nodes):
            mfpt[i, j] = 1. / peq[j, 0]*(q_inv[j, j]-q_inv[i, j]
                                         + identity[i, j])
    return mfpt


def compute_mfpt_rate(peq, rate):
    """
    Calculates the mean first passage times of the rate matrix.

    Parameters
    ----------
    peq : torch.Tensor(n_nodes)
        Eigenvector of the stationary distribution.
    rate : torch.Tensor(n_nodes,n_nodes)
        Rate matrix.

    Returns
    -------
    mfpt : torch.Tensor(n_nodes,n_nodes)
        Mean first passage time matrix.
    """
    n_nodes = peq.shape[0]
    onevec = torch.ones((1, n_nodes), dtype=torch.float64)
    peq = peq.reshape(peq.shape[0], 1)

    q_inv = torch.linalg.inv(torch.matmul(peq, onevec)-rate)
    mfpt = torch.zeros((n_nodes, n_nodes), dtype=torch.float64)

    for j in range(n_nodes):
        for i in range(n_nodes):
            mfpt[i, j] = 1. / peq[j, 0]*(q_inv[j, j]-q_inv[i, j])

    return mfpt


def create_rate_one_dim(potential, n_nodes, beta, exp_pre=10.):
    """
    Create rate matrix from potential.

    Parameters
    ----------
    potential : function
        Potential.
    n_bins : int
        Number of bins.
    beta : float
        Thermodynamic beta.
    exp_pre : float
        Exponential pre-factor.

    Notes
    -----
    This function constructs the rate matrix using physicist's notation.
    This means that columns are given by the first index and rows by the
    second.
    The rate matrix needs to be transposed so that it is represented in
    standard notation.

    Returns
    -------
    rate : torch.Tensor(n_nodes, n_nodes)
        Rate matrix.
    """

    # Map the continuous motion in 1D to a Markov process on a linear chain.
    # Discretize the configuration space in N bins.
    # K_{ji}=0 for j \neq i \pm 1
    rate = torch.zeros((n_nodes, n_nodes), dtype=torch.float64)

    for i in range(n_nodes-1):
        # Equation (41)torch
        rate[i, i + 1] = exp_pre * torch.exp(- beta * (potential[i]
                                                       - potential[i + 1]) / 2)
        rate[i + 1, i] = exp_pre * torch.exp(- beta * (potential[i + 1]
                                                       - potential[i]) / 2)
    for j in range(n_nodes):
        rate[j, j] = 0
        rate[j, j] = - torch.sum(rate[:, j])

    return rate


def ke(mfpt, peq):
    n_nodes, n_nodes = mfpt.shape

    ki = torch.zeros(n_nodes)
    for i in range(n_nodes):
        for j in range(n_nodes):
            ki[i] = ki[i] + mfpt[i, j] * peq[j, 0]

    return ki


def calc_d_pi(peq):
    r"""
    Computes the D_{\pi} matrix.

    Parameters
    ----------
    peq : torch.Tensor(n_nodes)
        Eigenvector

    Returns
    -------
    D_pi : torch.Tensor(n_nodes, n_nodes)
        D_pi matrix. Diagonal matrix with peq entries in the diagonal.
    """
    D_pi = torch.diag(peq[:, 0])

    return D_pi


def calc_inv_rate(rate):
    """
    Calculate the inverse of the rate matrix.

    Parameters
    ----------
    rate : torch.Tensor(n_nodes, n_nodes)
        Rate matrix.

    Returns
    -------
    inv_rate : torch.Tensor(n_nodes, n_nodes)
        Inverse of the rate matrix.
    """
    n_nodes = rate.shape[1]
    onevec = torch.ones(1, n_nodes, dtype=torch.float64)

    # Compute eigenvectors (already normalized) and eigenvalues
    eigenvalues, eigenvectors = linalg.eig(rate)

    # Sort the eigenvalues in descending order
    eigenvalues_sorted, indices = torch.sort(eigenvalues.real, descending=True)

    # Normalise the matrix so that \sum_j M_{ij} = 1
    eigenvectors = eigenvectors / torch.sum(eigenvectors, dim=0)

    # Get the stationary probability (or equilibrium probability)
    # Corresponds to the eigenvector with eigenvalue zero
    peq = eigenvectors[:, indices[0]].real
    peq = peq.reshape(n_nodes, 1)

    # Invert matrix
    inv_rate = linalg.inv(peq @ onevec - rate)

    return inv_rate, peq, eigenvalues_sorted


def Markov_HS_clusteringS(M, peq, S):
    """
    TODO: Add docstring
    """
    N, _ = M.shape
    n_clusters = S.shape[0]
    P_EQ = S @ peq.T

    ONE_VEC = np.ones((1, n_clusters))
    I_n = np.diag(ONE_VEC[0])

    one_vec = np.ones((1, N))
    I_N = np.diag(one_vec[0])

    D_N = np.diag(P_EQ)
    D_n = np.diag(peq)

    inv_1 = inv(I_N + peq.T * one_vec - M.T)
    inv_1 = S @ inv_1
    R = (I_n + P_EQ*ONE_VEC -
         D_N*(inv(inv_1 * D_n @ S.T)))
    return R

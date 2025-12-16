"""
Graph Utilities Module

This module contains functions for generating graphs and initializing couplings
for quantum spin glass simulations.

Source files:
- MF susc prog.ipynb (Cell 1)
- Cavity + Susc prop.ipynb (Cell 1)
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple


def generate_random_regular_graph(N: int, degree: int, seed: int = None) -> nx.Graph:
    """
    Build a random regular graph with N nodes and fixed degree.
    
    A random regular graph is a graph where every node has exactly the same
    number of neighbors. This is used to model mean-field spin glass systems.
    
    Parameters:
    -----------
    N : int
        Number of nodes (spins)
    degree : int
        Degree of each node (coordination number + 1)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    nx.Graph
        NetworkX graph object
    
    Source:
    -------
    MF susc prog.ipynb, Cell 1
    Cavity + Susc prop.ipynb, Cell 1
    """
    return nx.random_regular_graph(d=degree, n=N, seed=seed)


def initialize_couplings(
    G: nx.Graph,
    std: float = 1.0,
    seed: int = None
) -> Dict[Tuple[int, int], float]:
    """
    Assign Gaussian couplings J_{ij} ~ Normal(0, std^2) to each undirected edge.
    
    The couplings represent the Ising interaction strengths between spins.
    They are drawn from a Gaussian distribution with zero mean and standard
    deviation std. The key format is (min(i,j), max(i,j)) to ensure uniqueness.
    
    Parameters:
    -----------
    G : nx.Graph
        NetworkX graph object
    std : float, default=1.0
        Standard deviation of Gaussian distribution for couplings
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    Dict[Tuple[int, int], float]
        Dictionary mapping edge tuples (i, j) with i < j to coupling values
    
    Source:
    -------
    MF susc prog.ipynb, Cell 1
    Cavity + Susc prop.ipynb, Cell 1
    """
    rng = np.random.default_rng(seed)
    J = {}
    for i, j in G.edges():
        key = (i, j) if i < j else (j, i)
        J[key] = rng.normal(loc=0.0, scale=std)
    return J


def couplings_dict_to_matrix(J: Dict[Tuple[int, int], float], G: nx.Graph) -> np.ndarray:
    """
    Build the symmetric coupling matrix J_mat (N x N) from the dictionary J.
    
    This converts the edge-based dictionary representation to a full matrix
    representation, which is useful for certain matrix-based computations.
    
    Parameters:
    -----------
    J : Dict[Tuple[int, int], float]
        Dictionary mapping (i, j) with i < j to coupling value
    G : nx.Graph
        NetworkX Graph (to determine node count)
    
    Returns:
    --------
    np.ndarray
        Symmetric coupling matrix of shape (N, N) with J_mat[i, j] = J_mat[j, i] = J[(min(i,j), max(i,j))]
        and zeros elsewhere
    
    Source:
    -------
    MF susc prog.ipynb, Cell 2
    """
    N = G.number_of_nodes()
    J_mat = np.zeros((N, N))
    for (i, j), value in J.items():
        J_mat[i, j] = value
        J_mat[j, i] = value
    return J_mat


def build_M(
    J_mat: np.ndarray,
    G: nx.Graph,
    _Gamma: float,
    K: int
) -> np.ndarray:
    """
    Construct the (2E)×(2E) matrix M, where E = number of undirected edges in G.
    Directed edges are ordered as (u→v), (v→u) for each undirected {u,v}.

    M_{(i→j),(m→n)} = - J_{i,m} / (Gamma * (K/(K+1)))  if n == i and (m != j)
                    = 0 otherwise

    In other words, we exclude couplings from (i→j) to (j→i).

    Source: MF susc prog-Copy1.ipynb, Cell 6
    """
    directed_edges = []
    for (u, v) in G.edges():
        directed_edges.append((u, v))
        directed_edges.append((v, u))
    size = len(directed_edges)

    dir_edge_index = {edge: idx for idx, edge in enumerate(directed_edges)}

    M = np.zeros((size, size))

    for row_idx, (i, j) in enumerate(directed_edges):
        for m in G.neighbors(i):
            if m == j:
                continue  # Skip inverse edge (j → i)
            col_edge = (m, i)
            col_idx = dir_edge_index[col_edge]
            Jij = J_mat[i, m]
            M[row_idx, col_idx] = Jij / (_Gamma * (K / (K + 1)))

    return M


def build_A(G: nx.Graph, K: int) -> np.ndarray:
    """
    Construct the matrix A of size (2E) x N, where E is the number of undirected edges
    in G and N is the number of nodes.

    A_{(i->j), n} = (K / (K + 1)) * δ_{n, i}

    Source: MF susc prog-Copy1.ipynb, Cell 7
    """
    N = G.number_of_nodes()
    directed_edges = []
    for (u, v) in G.edges():
        directed_edges.append((u, v))
        directed_edges.append((v, u))

    num_rows = len(directed_edges)
    A = np.zeros((num_rows, N))

    factor = K / (K + 1)

    for row_idx, (i, j) in enumerate(directed_edges):
        A[row_idx, i] = factor

    return A


def build_B(J_mat: np.ndarray, G: nx.Graph, _Gamma: float, K: int) -> np.ndarray:
    """
    Construct the matrix B of size N x (2E), where E is the number of undirected edges
    in G and N is the number of nodes.

    B_{i,(m->n)} = - J_{i,m}/(Gamma * (K/(K+1))) * δ_{i,n}

    Source: MF susc prog-Copy1.ipynb, Cell 8
    """
    N = G.number_of_nodes()
    directed_edges = []
    for (u, v) in G.edges():
        directed_edges.append((u, v))
        directed_edges.append((v, u))

    num_cols = len(directed_edges)
    B = np.zeros((N, num_cols))

    for col_idx, (m, n) in enumerate(directed_edges):
        B[n, col_idx] = J_mat[n, m] / (_Gamma * K/(K+1))

    return B


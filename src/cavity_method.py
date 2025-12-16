"""
Cavity Method (Belief Propagation) Module

This module implements the cavity method, also known as belief propagation,
for quantum spin glasses. It includes functions for initializing messages,
running belief propagation iterations, and computing cavity message updates.

Source files:
- Cavity + Susc prop.ipynb (Cells 1-2, 4-5)
- MF susc prog.ipynb (Cells 29-31)
"""

import numpy as np
import networkx as nx
from typing import Callable, Dict, Tuple
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

from .quantum_operators import make_expH, make_sx0, make_sz0, full_mess
from .graph_utils import generate_random_regular_graph, initialize_couplings

# Type aliases
directed_edge = Tuple[int, int]
MessageDict = Dict[directed_edge, float]


def initialize_messages(
    G: nx.Graph,
    transverse_field: float,
    std: float = 0.1,
    seed: int = None
) -> Tuple[MessageDict, MessageDict]:
    """
    Initialize directed cavity messages h_{i->j} and b_{i->j} to small random values.
    
    The messages are initialized with small random Gaussian noise. The b messages
    are offset by the transverse field value to provide a reasonable starting point.
    
    Parameters:
    -----------
    G : nx.Graph
        NetworkX graph object
    transverse_field : float
        Transverse field strength (used to offset b messages)
    std : float, default=0.1
        Standard deviation for initialization
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    tuple (MessageDict, MessageDict)
        Initialized message dictionaries (h, b) for all directed edges
    
    Source:
    -------
    Cavity + Susc prop.ipynb, Cell 1
    """
    rng = np.random.default_rng(seed)
    h = {}
    b = {}
    for i, j in G.edges():
        h[(i, j)] = rng.normal(0.0, std)
        h[(j, i)] = rng.normal(0.0, std)
        b[(i, j)] = rng.normal(0.0, std) + transverse_field
        b[(j, i)] = rng.normal(0.0, std) + transverse_field
    return h, b


def F_func(neighbor_hs, neighbor_bs, neighbor_couplings, beta, transverse_field):
    """
    Quantum cavity message update function.
    
    This function computes the new cavity messages (h, b) for a site given the
    messages from its neighbors. It uses the full quantum cavity method with
    matrix exponentials.
    
    Parameters:
    -----------
    neighbor_hs : list or np.ndarray
        Longitudinal cavity fields h from neighbors
    neighbor_bs : list or np.ndarray
        Transverse cavity fields b from neighbors
    neighbor_couplings : list or np.ndarray
        Coupling strengths J to neighbors
    beta : float
        Inverse temperature
    transverse_field : float
        Transverse field strength
    
    Returns:
    --------
    tuple (float, float)
        Updated cavity messages (new_h, new_b)
    
    Source:
    -------
    Cavity + Susc prop.ipynb, Cell 4
    """
    K_val = len(neighbor_hs)
    
    # Convert to numpy arrays if needed
    if isinstance(neighbor_hs, list):
        neighbor_hs = np.array(neighbor_hs)
    if isinstance(neighbor_bs, list):
        neighbor_bs = np.array(neighbor_bs)
    if isinstance(neighbor_couplings, list):
        neighbor_couplings = np.array(neighbor_couplings)
    
    sx0 = make_sx0(K_val)
    sz0 = make_sz0(K_val)
    
    # Compute the matrix exponential
    expmat = make_expH(beta, K_val, neighbor_couplings, transverse_field, neighbor_hs, neighbor_bs)
    
    # Get new message from cavity magnetizations
    new_h, new_b = full_mess(beta, sz0, sx0, expmat)
    
    return new_h, new_b


def belief_propagation(
    G: nx.Graph,
    J: Dict[Tuple[int, int], float],
    F_func: Callable,
    beta: float,
    transverse_field: float,
    max_iter: int = 1000,
    tol: float = 1e-4,
    init_std: float = 0.01,
    seed: int = None
) -> Tuple[MessageDict, MessageDict, bool, int]:
    """
    Run belief propagation (cavity method) until convergence.
    
    This function iteratively updates cavity messages until either:
    1. Convergence is reached (max_diff < tol)
    2. RSB (Replica Symmetry Breaking) is detected (no decrease in error)
    3. Maximum iterations are reached
    
    Parameters:
    -----------
    G : nx.Graph
        NetworkX graph object
    J : Dict[Tuple[int, int], float]
        Dictionary of coupling strengths
    F_func : Callable
        Message update function (typically F_func from this module)
    beta : float
        Inverse temperature
    transverse_field : float
        Transverse field strength
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-4
        Convergence tolerance
    init_std : float, default=0.01
        Standard deviation for message initialization
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    tuple (MessageDict, MessageDict, bool, int)
        Converged messages (h, b), RSB flag, and number of iterations
    
    Source:
    -------
    Cavity + Susc prop.ipynb, Cell 2
    """
    h, b = initialize_messages(G, transverse_field, std=init_std, seed=seed)
    edges = list(h.keys())
    max_diffs = []
    rsb_flag = False

    for it in range(1, max_iter + 1):
        max_diff = 0.0
        new_h, new_b = {}, {}

        # One full sweep over all directed edges
        for (i, j) in edges:
            # Gather neighbor messages (excluding j)
            neigh = [k for k in G.neighbors(i) if k != j]
            J_arr = np.array([J[(k, i) if k < i else (i, k)] for k in neigh])
            h_arr = np.array([h[(k, i)] for k in neigh])
            b_arr = np.array([b[(k, i)] for k in neigh])
            
            # Update message using F_func
            upd_h, upd_b = F_func(h_arr.tolist(), b_arr.tolist(), J_arr, beta, transverse_field)
            new_h[(i, j)] = upd_h
            new_b[(i, j)] = upd_b
            
            # Track maximum change
            max_diff = max(max_diff, abs(upd_h - h[(i, j)]), abs(upd_b - b[(i, j)]))

        # Atomic update (update all messages at once)
        h.update(new_h)
        b.update(new_b)
        max_diffs.append(max_diff)

        # Progress reporting
        if (it % 100) == 0:
            print(it)
            print(max_diff)

        # Check for strict convergence
        if max_diff < tol:
            return h, b, False, it

        # Check RSB-style no-decrease after 50 iterations
        if it >= 300 and (it % 50) == 0:
            prev_avg = np.mean(max_diffs[-100:-50])
            last_avg = np.mean(max_diffs[-50:])
            if last_avg >= prev_avg:
                rsb_flag = True
                return h, b, True, it

    # Hit max_iter without RSB-stop
    return h, b, False, it


def compute_susceptibility_prop(
    G: nx.Graph,
    J: Dict[Tuple[int, int], float],
    h_conv: MessageDict,
    b_conv: MessageDict,
    beta: float,
    transverse_field: float
) -> Tuple[complex, np.ndarray]:
    """
    Compute susceptibility propagation: leading eigenvalue of the Jacobian matrix.
    
    This function computes the stability of the converged belief propagation solution
    by finding the leading eigenvalue of the Jacobian matrix of the message update
    function. If |lambda_max| > 1, the solution is unstable (RSB phase).
    
    Parameters:
    -----------
    G : nx.Graph
        NetworkX graph object
    J : Dict[Tuple[int, int], float]
        Dictionary of coupling strengths
    h_conv : MessageDict
        Converged longitudinal cavity messages
    b_conv : MessageDict
        Converged transverse cavity messages
    beta : float
        Inverse temperature
    transverse_field : float
        Transverse field strength
    
    Returns:
    --------
    tuple (complex, np.ndarray)
        Leading eigenvalue and corresponding eigenvector
    
    Source:
    -------
    Cavity + Susc prop.ipynb, Cell 5
    """
    # 1) Enumerate messages and build index maps
    directed_edges = sorted(h_conv.keys())  # list of (i,j)
    E = len(directed_edges)
    M = 2 * E  # total scalar messages (h and b for each edge)

    # Map (edge, field_type) to index: field_type 0 for h, 1 for b
    msg_to_idx = {}
    for idx, edge in enumerate(directed_edges):
        msg_to_idx[(edge, 0)] = 2 * idx
        msg_to_idx[(edge, 1)] = 2 * idx + 1
    
    # Precompute dependencies: for each message (p->q), the set of edges q->r (r != p)
    deps = {}  # key: (edge, field_type) for input, value: list of (dep_edge, dep_field)
    for edge in directed_edges:
        p, q = edge
        # Perturbing either h or b on (p->q) affects all updates for edges (q->r)
        out_edges = [(q, r) for r in G.neighbors(q) if r != p]
        for ft in (0, 1):  # field type of input
            key = (edge, ft)
            # For each dependent directed_edge, two output types
            deps[key] = []
            for e_out in out_edges:
                deps[key].append((e_out, 0))  # h-output at e_out
                deps[key].append((e_out, 1))  # b-output at e_out

    # 2) Compute base sweep outputs
    def single_sweep_dict(h_dict, b_dict):
        """Single-sweep update for all edges."""
        out_h = {}
        out_b = {}
        for (i, j) in directed_edges:
            # Gather neighbor messages
            nh, nb, jc = [], [], []
            for k in G.neighbors(i):
                if k == j:
                    continue
                nh.append(h_dict[(k, i)])
                nb.append(b_dict[(k, i)])
                key = (k, i) if k < i else (i, k)
                jc.append(J[key])
            out_h[(i, j)], out_b[(i, j)] = F_func(nh, nb, jc, beta, transverse_field)
        return out_h, out_b

    F0_h, F0_b = single_sweep_dict(h_conv, b_conv)

    # 3) Build sparse Jacobian by finite differences
    eps = 1e-7
    J_mat = sp.lil_matrix((M, M))

    # For each input message coordinate
    for (edge_in, ft), col in msg_to_idx.items():
        # Original value
        orig_val = h_conv[edge_in] if ft == 0 else b_conv[edge_in]
        # Perturbed copy of that single message
        pert_value = orig_val + eps
        
        # Iterate dependencies
        for (edge_out, out_ft) in deps[(edge_in, ft)]:
            i, j = edge_out
            # Build neighbor lists for update (i->j) using conv values but replacing the one perturbed
            nh, nb, jc = [], [], []
            for k in G.neighbors(i):
                if k == j:
                    continue
                # Choose perturbed if (k,i)==edge_in
                if (k, i) == edge_in:
                    h_val = pert_value if ft == 0 else h_conv[(k, i)]
                    b_val = pert_value if ft == 1 else b_conv[(k, i)]
                else:
                    h_val = h_conv[(k, i)]
                    b_val = b_conv[(k, i)]
                nh.append(h_val)
                nb.append(b_val)
                key = (k, i) if k < i else (i, k)
                jc.append(J[key])
            
            # Compute only this one update
            upd_h, upd_b = F_func(nh, nb, jc, beta, transverse_field)
            # Base output
            base_val = F0_h[edge_out] if out_ft == 0 else F0_b[edge_out]
            diff = (upd_h - base_val) if out_ft == 0 else (upd_b - base_val)
            row = msg_to_idx[(edge_out, out_ft)]
            J_mat[row, col] = diff / eps

    J_csr2 = J_mat.tocsr()

    # 4) Compute leading eigenpair
    vals2, vecs2 = eigs(J_csr2, k=1, which='LM')
    lambda_max2 = vals2[0]
    v_max2 = vecs2[:, 0]

    return lambda_max2, v_max2


"""
Quantum Operators Module

This module contains functions for constructing quantum operators used in the
cavity method for quantum spin glasses. It includes Pauli matrices, effective
Hamiltonian construction, and expectation value computations.

Source files:
- Pop_dyn_Q-ISING_GAUSS-POPGEN.ipynb (Cells 3-9)
- Cavity + Susc prop.ipynb (Cell 3)
"""

import numpy as np
from scipy.linalg import expm

# Pauli matrices
sx = np.array([[0, 1], [1, 0]])  # Pauli-X
sy = np.array([[0, -1j], [1j, 0]])  # Pauli-Y
sz = np.array([[1, 0], [0, -1]])  # Pauli-Z

# Identity matrix
sid = np.eye(2)  # 2x2 identity matrix


def make_betaHeff(beta, K, J_arr, G, h_arr, b_arr):
    """
    Construct the effective Hamiltonian -beta*H_eff for a cavity site with K neighbors.
    
    The effective Hamiltonian includes:
    - Transverse field terms on all sites
    - Ising interactions between the cavity site and its neighbors
    - Cavity field contributions (h_arr for longitudinal, b_arr for transverse)
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    K : int
        Coordination number (number of neighbors)
    J_arr : np.ndarray
        Array of coupling strengths J_i for each neighbor
    G : float
        Transverse field strength
    h_arr : np.ndarray
        Array of longitudinal cavity fields h_i from each neighbor
    b_arr : np.ndarray
        Array of transverse cavity fields b_i from each neighbor
    
    Returns:
    --------
    np.ndarray
        Matrix representation of -beta * H_eff (shape: 2^(K+1) x 2^(K+1))
    
    Source:
    -------
    Pop_dyn_Q-ISING_GAUSS-POPGEN.ipynb, Cell 4
    Cavity + Susc prop.ipynb, Cell 3
    """
    sx_list = []
    sy_list = []
    sz_list = []

    # Construct sx_n, sy_n, sz_n for each site n
    for n in range(K+1):
        op_list = [sid] * (K+1)  # Start with identity for all sites

        # Construct sx_n
        op_list[n] = sx
        sx_n = op_list[0]
        for m in range(1, K+1):
            sx_n = np.kron(sx_n, op_list[m])
        sx_list.append(sx_n)

        # Construct sy_n
        op_list[n] = sy
        sy_n = op_list[0]
        for m in range(1, K+1):
            sy_n = np.kron(sy_n, op_list[m])
        sy_list.append(sy_n)

        # Construct sz_n
        op_list[n] = sz
        sz_n = op_list[0]
        for m in range(1, K+1):
            sz_n = np.kron(sz_n, op_list[m])
        sz_list.append(sz_n)

    # Construct H0
    # Transverse field on cavity site (site 0)
    H0 = K * G / (K + 1) * sx_list[0]

    # Interactions and fields on neighbors
    for n in range(1, K+1):
        # Ising interaction between cavity site and neighbor n
        H0 += J_arr[n-1] * (sz_list[0] @ sz_list[n])
        # Transverse field on neighbor n
        H0 += G / (K + 1) * sx_list[n]
        # Cavity fields on neighbor n
        H0 += b_arr[n-1] * sx_list[n] + h_arr[n-1] * sz_list[n]

    return -beta * H0


def make_expH(beta, K, J_arr, G, h_arr, b_arr):
    """
    Compute the matrix exponential exp(-beta * H_eff).
    
    This is the Boltzmann weight operator used in quantum cavity calculations.
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    K : int
        Coordination number
    J_arr : np.ndarray
        Array of coupling strengths
    G : float
        Transverse field strength
    h_arr : np.ndarray
        Array of longitudinal cavity fields
    b_arr : np.ndarray
        Array of transverse cavity fields
    
    Returns:
    --------
    np.ndarray
        Matrix exponential exp(-beta * H_eff)
    
    Source:
    -------
    Pop_dyn_Q-ISING_GAUSS-POPGEN.ipynb, Cell 5
    Cavity + Susc prop.ipynb, Cell 3
    """
    betaHeff = make_betaHeff(beta, K, J_arr, G, h_arr, b_arr)
    return expm(betaHeff)


def make_sx0(K):
    """
    Construct the sx operator for site 0 (cavity site) in a system of K+1 sites.
    
    Parameters:
    -----------
    K : int
        Coordination number (number of neighbors)
    
    Returns:
    --------
    np.ndarray
        Matrix representation of sx_0 (shape: 2^(K+1) x 2^(K+1))
    
    Source:
    -------
    Pop_dyn_Q-ISING_GAUSS-POPGEN.ipynb, Cell 6
    Cavity + Susc prop.ipynb, Cell 3
    """
    op_list = [sid] * (K + 1)  # Start with identity for all sites
    op_list[0] = sx  # Replace the first site with sx

    # Compute the Kronecker product of all operators in the list
    result = op_list[0]
    for m in range(1, K + 1):
        result = np.kron(result, op_list[m])

    return result


def make_sz0(K):
    """
    Construct the sz operator for site 0 (cavity site) in a system of K+1 sites.
    
    Parameters:
    -----------
    K : int
        Coordination number (number of neighbors)
    
    Returns:
    --------
    np.ndarray
        Matrix representation of sz_0 (shape: 2^(K+1) x 2^(K+1))
    
    Source:
    -------
    Pop_dyn_Q-ISING_GAUSS-POPGEN.ipynb, Cell 7
    Cavity + Susc prop.ipynb, Cell 3
    """
    op_list = [sid] * (K + 1)  # Start with identity for all sites
    op_list[0] = sz  # Replace the first site with sz

    # Compute the Kronecker product of all operators in the list
    result = op_list[0]
    for m in range(1, K + 1):
        result = np.kron(result, op_list[m])

    return result


def expect_val(op, expmat):
    """
    Compute the expectation value of an operator with respect to the density matrix.
    
    The expectation value is computed as Tr(op @ expmat) / Tr(expmat), where
    expmat is the unnormalized density matrix exp(-beta * H_eff).
    
    Parameters:
    -----------
    op : np.ndarray
        Operator matrix
    expmat : np.ndarray
        Unnormalized density matrix (matrix exponential)
    
    Returns:
    --------
    float
        Expectation value (real part only)
    
    Source:
    -------
    Pop_dyn_Q-ISING_GAUSS-POPGEN.ipynb, Cell 8
    Cavity + Susc prop.ipynb, Cell 3
    """
    # Compute the numerator: trace(op @ expmat)
    numerator = np.trace(op @ expmat)

    # Compute the denominator: trace(expmat)
    denominator = np.trace(expmat)

    # Return the real part of the normalized expectation value
    return np.real(numerator / denominator)


def full_mess(beta, sz0, sx0, expmat):
    """
    Compute the full cavity message (h, b) from the expectation values.
    
    This function extracts the cavity fields h (longitudinal) and b (transverse)
    from the expectation values of sz and sx operators. The message update rule
    is derived from the cavity method for quantum spin glasses.
    
    Parameters:
    -----------
    beta : float
        Inverse temperature
    sz0 : np.ndarray
        sz operator for the cavity site
    sx0 : np.ndarray
        sx operator for the cavity site
    expmat : np.ndarray
        Matrix exponential exp(-beta * H_eff)
    
    Returns:
    --------
    tuple (float, float)
        Updated cavity messages (h_mess, b_mess)
    
    Source:
    -------
    Pop_dyn_Q-ISING_GAUSS-POPGEN.ipynb, Cell 9
    Cavity + Susc prop.ipynb, Cell 3
    """
    mz0 = expect_val(sz0, expmat)
    mx0 = expect_val(sx0, expmat)

    # Handle the case where both magnetizations are zero
    if (mx0**2 + mz0**2) == 0:
        h_mess = -1/beta * mz0
        b_mess = -1/beta * mx0
    else:
        # Standard message update rule
        h_mess = -1/beta * mz0 * np.arctanh(np.sqrt(mx0**2 + mz0**2)) / np.sqrt(mx0**2 + mz0**2)
        b_mess = -1/beta * mx0 * np.arctanh(np.sqrt(mx0**2 + mz0**2)) / np.sqrt(mx0**2 + mz0**2)

    return h_mess, b_mess


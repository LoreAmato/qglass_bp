"""
Analysis Utilities Module

This module contains functions for analyzing population dynamics results,
including loading saved data, computing statistics, and processing parameter sweeps.

Source files:
- Q-analysis_GAUSS.ipynb (Cells 1-3, 9-15)
- Pop_dyn_Q-ISING_GAUSS-POPGEN.ipynb (Cells 14-15)
"""

import numpy as np
import os
import pickle
import itertools
from numpy.linalg import inv
from scipy.sparse.linalg import eigs
from scipy.stats import iqr

from .graph_utils import build_M, build_A, build_B, generate_random_regular_graph, initialize_couplings, couplings_dict_to_matrix


# ---- Basic helpers ----

def gen_ipr(psi, q):
    """
    Generalized inverse participation ratio.

    Source: MF susc prog-Copy1.ipynb, Cell 1
    """
    return np.sum(np.abs(psi)**(2*q)) / np.sum(np.abs(psi)**2)


# ---- Plot helpers (kept close to original styling) ----
def setup_plotting():
    """
    Configure matplotlib for publication-quality figures.

    Source: MF susc prog-Copy1.ipynb, Cell 2
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Computer Modern Roman'],
        'font.size': 12,
        'mathtext.fontset': 'stix',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (6, 4.5),
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'grid.color': '#dddddd',
        'grid.linestyle': '--',
        'grid.linewidth': 0.7,
        'grid.alpha': 0.6,
        'legend.fontsize': 11,
        'legend.framealpha': 0.8,
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        'lines.linewidth': 2.0,
        'lines.markersize': 7,
        'errorbar.capsize': 3.5,
    })
    try:
        plt.rcParams.update({
            'text.usetex': True,
            'text.latex.preamble': r'\\usepackage{amsmath, amssymb}',
        })
    except Exception:
        plt.rcParams.update({'text.usetex': False})
    return plt


def create_figure():
    """
    Create a pre-configured figure and axes with consistent styling.

    Source: MF susc prog-Copy1.ipynb, Cell 3
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.tick_params(axis='both', which='both', top=True, right=True)
    ax.tick_params(axis='both', which='minor', width=0.75, length=2.5)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.2)
    return fig, ax


def save_figure(filename, formats=('pdf', 'png'), transparent=False):
    """
    Save current figure in multiple formats.

    Source: MF susc prog-Copy1.ipynb, Cell 4
    """
    import matplotlib.pyplot as plt
    for fmt in formats:
        fullname = f"{filename}.{fmt}"
        plt.savefig(
            fullname,
            format=fmt,
            bbox_inches='tight',
            pad_inches=0.05,
            transparent=transparent,
            facecolor='white' if not transparent else 'none'
        )
    plt.show()


def compute_from_population_dynamics(param_dict, compute_func, data_dir="Qtest_results"):
    """
    Compute a function on population dynamics results for all parameter combinations.
    
    This function loads saved population dynamics results from pickle files and
    applies a computation function to each result. The results are organized by
    parameter combinations.
    
    Parameters:
    -----------
    param_dict : dict
        Dictionary where keys are parameter names and values are arrays of parameter values
    compute_func : callable
        Function that takes a results dictionary and returns a computed value
    data_dir : str, default="Qtest_results"
        Directory where .pkl files are stored
    
    Returns:
    --------
    dict
        Dictionary where keys are tuples of parameter values and values are computed outputs
    
    Source:
    -------
    Q-analysis_GAUSS.ipynb, Cell 1
    """
    results_dict = {}

    # Generate all combinations of parameter values
    param_names = list(param_dict.keys())
    param_combinations = list(itertools.product(*param_dict.values()))

    # Iterate over all parameter combinations
    for param_values in param_combinations:
        # Construct the parameter dictionary for this combination
        params = dict(zip(param_names, param_values))

        # Generate the corresponding filename
        filename = f"Q-results_N{params['N']}_sigma{params['sigma']}_K{params['K']}_beta{params['beta']}_G{params['G']}.pkl"
        file_path = os.path.join(data_dir, filename)

        # Check if file exists before proceeding
        if not os.path.exists(file_path):
            print(f"Warning: File {filename} not found. Skipping...")
            continue

        # Load the file and extract results
        with open(file_path, "rb") as file:
            results = pickle.load(file)

        # Compute the desired quantity
        computed_value = compute_func(results)

        # Store results in dictionary with parameter tuple as key
        results_dict[param_values] = computed_value

    return results_dict


def extract_parameters_from_pkl(data_dir="Qtest_results"):
    """
    Extract and organize parameters from all .pkl files in a directory.
    
    This function scans a directory for pickle files containing population dynamics
    results and extracts the parameter sets used in each simulation.
    
    Parameters:
    -----------
    data_dir : str, default="Qtest_results"
        Directory where .pkl files are stored
    
    Returns:
    --------
    tuple (dict, list)
        - param_dict: Dictionary where keys are parameter names and values are
          unique sorted arrays of parameter values
        - param_list: List of dictionaries containing parameter sets for each pkl file
    
    Source:
    -------
    Q-analysis_GAUSS.ipynb, Cell 3
    """
    param_list = []

    # Iterate over all .pkl files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith(".pkl"):
            file_path = os.path.join(data_dir, filename)
            
            # Load only the "parameters" entry
            with open(file_path, "rb") as file:
                results = pickle.load(file)
                params = results["parameters"]  # Extract parameters dictionary
                param_list.append(params)

    # Organize parameters into separate arrays
    param_dict = {}
    for param_set in param_list:
        for key, value in param_set.items():
            if key not in param_dict:
                param_dict[key] = []
            param_dict[key].append(value)

    # Convert lists to unique sorted numpy arrays
    for key in param_dict:
        param_dict[key] = np.array(sorted(set(param_dict[key])))

    return param_dict, param_list


def convert_results_to_arrays(results_dict, param_names):
    """
    Convert a dictionary of results into structured NumPy arrays.
    
    This function organizes results from a parameter sweep into a multi-dimensional
    array structure, allowing array-like indexing by parameter values.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys as tuples of parameters and values as computed results
        (which can be arrays)
    param_names : list
        List of parameter names in the tuple keys
    
    Returns:
    --------
    tuple (dict, np.ndarray)
        - param_values: Dictionary with parameter names as keys and sorted arrays as values
        - results_array: Multi-dimensional object array containing computed results
    
    Source:
    -------
    Q-analysis_GAUSS.ipynb, Cell 9
    """
    # Extract unique parameter values
    param_values = {
        name: sorted(set(param[i] for param in results_dict.keys()))
        for i, name in enumerate(param_names)
    }

    # Initialize results array as an object array
    results_shape = tuple(len(param_values[name]) for name in param_names)
    results_array = np.empty(results_shape, dtype=object)

    # Fill the results array
    for param_tuple, result in results_dict.items():
        indices = tuple(
            param_values[name].index(param_tuple[i])
            for i, name in enumerate(param_names)
        )
        results_array[indices] = np.array(result)

    return param_values, results_array


# Helper functions for extracting specific quantities from results
def get_pop_h(result):
    """Extract h population from results dictionary."""
    return result["pop_h"]


def get_pop_b(result):
    """Extract b population from results dictionary."""
    return result["pop_b"]


def get_stds_h(result):
    """Extract h standard deviations time series from results dictionary."""
    return result["stds_h"]


def get_stds_b(result):
    """Extract b standard deviations time series from results dictionary."""
    return result["stds_b"]


def get_means_h(result):
    """Extract h means time series from results dictionary."""
    return result["means_h"]


def get_means_b(result):
    """Extract b means time series from results dictionary."""
    return result["means_b"]


def get_iqrs_h(result):
    """Extract h IQRs time series from results dictionary."""
    return result["iqrs_h"]


def get_iqrs_b(result):
    """Extract b IQRs time series from results dictionary."""
    return result["iqrs_b"]


def get_medians_h(result):
    """Extract h medians time series from results dictionary."""
    return result["medians_h"]


def get_medians_b(result):
    """Extract b medians time series from results dictionary."""
    return result["medians_b"]


def mean_excess_function(data, s_values, min_exceedances=10):
    """
    Compute the Mean Excess Function (MEF) for a range of thresholds.
    
    The mean excess function is used in extreme value theory to characterize
    the tail behavior of distributions. For a threshold s, it computes the
    mean of exceedances above s.
    
    Parameters:
    -----------
    data : np.ndarray
        1D array of positive data values
    s_values : array-like
        Array of candidate threshold values
    min_exceedances : int, default=10
        Minimum number of exceedances required to compute the MEF
    
    Returns:
    --------
    tuple (np.ndarray, np.ndarray)
        - valid_s: Thresholds for which MEF was computed
        - e_s: Corresponding mean excess values
    
    Source:
    -------
    Pop_dyn_Q-ISING_GAUSS-POPGEN.ipynb, Cell 15
    """
    valid_s = []
    e_s = []
    
    for s in s_values:
        # Select exceedances above s
        exceedances = data[data > s]
        k = len(exceedances)
        if k < min_exceedances:
            # Skip thresholds with too few exceedances
            continue
        
        # Compute mean excess function
        mean_excess = np.mean(exceedances - s)
        valid_s.append(s)
        e_s.append(mean_excess)
    
    return np.array(valid_s), np.array(e_s)


# ---- Critical Gamma and multifractal helpers ----

def invert_function(x, f_x):
    """
    Numerically invert a function given sampled pairs (x, f(x)).

    Source: MF susc prog-Copy1.ipynb, Cell 92
    """
    from scipy.interpolate import interp1d
    return interp1d(f_x, x, bounds_error=False)

def find_near_critical_Gamma(J_mat, G, K_val,
                             Gamma_start=1.5,
                             step=0.3,
                             tol_low=0.99,
                             tol_high=1.0,
                             max_iter=100):
    """
    Find Gamma such that the largest eigenvalue of M lies in [tol_low, tol_high).
    Uses an adaptive step-search: if lam < tol_low, decrease Gamma; if lam >= tol_high, increase Gamma.

    Source: MF susc prog-Copy1.ipynb, Cell 10
    """
    Gamma = Gamma_start
    direction = -1  # -1 decreasing Gamma, +1 increasing
    for _ in range(max_iter):
        M = build_M(J_mat, G, Gamma, K_val)
        vals, _ = eigs(M, k=30, which='LM')
        lam_max = np.real(vals).max()

        if tol_low < lam_max < tol_high:
            return Gamma, M

        direction = -1 if lam_max < tol_low else +1
        Gamma += direction * step
        step *= 0.9

    print(Gamma, lam_max)
    return Gamma, M


def compute_ipr_and_gamma_for_config(N, degree, std_J, q_vals, trial_seed):
    """
    One trial: graph generation, coupling init, critical Gamma search, susceptibility & IPR.

    Returns generalized IPR array over q_vals and the near-critical Gamma.

    Source: MF susc prog-Copy1.ipynb, Cell 10
    """
    G = generate_random_regular_graph(N, degree, seed=trial_seed)
    J_dict = initialize_couplings(G, std=std_J, seed=trial_seed + 99)
    J_mat = couplings_dict_to_matrix(J_dict, G)
    K_val = degree - 1

    Gamma_c, M_mat = find_near_critical_Gamma(J_mat, G, K_val)

    A_mat = build_A(G, K_val)
    B_mat = build_B(J_mat, G, Gamma_c, K_val)
    size = degree * N
    prop = inv(np.eye(size) - M_mat)

    Chi = (1.0 / Gamma_c) * (np.eye(N) + B_mat @ prop @ A_mat)
    vals, vecs = eigs(Chi, k=30, which='LR')
    leading_vec = vecs[:, np.argmax(np.real(vals))]

    gen_iprs = np.array([gen_ipr(leading_vec, q) for q in q_vals])
    return gen_iprs, Gamma_c


def run_experiments(N_list, degree, std_J, q_vals, trials=30, base_seed=92):
    """
    Run multiple trials per N; collect IPRs and critical Gammas.

    Returns:
      gen_iprs_results: dict mapping N -> list of IPR arrays (one per trial)
      gamma_results: dict mapping N -> list of Gamma_c values (one per trial)

    Source: MF susc prog-Copy1.ipynb, Cell 10
    """
    gen_iprs_results = {}
    gamma_results = {}
    for idx, N in enumerate(N_list):
        gen_iprs = []
        gammas = []
        for t in range(trials):
            seed = base_seed + idx * trials + t
            ipr_val, gamma_c = compute_ipr_and_gamma_for_config(N, degree, std_J, q_vals, seed)
            gen_iprs.append(ipr_val)
            gammas.append(gamma_c)
        gen_iprs_results[N] = gen_iprs
        gamma_results[N] = gammas
    return gen_iprs_results, gamma_results


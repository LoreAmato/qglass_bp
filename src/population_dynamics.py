"""
Population Dynamics Module

This module implements population dynamics for quantum spin glasses. Population
dynamics is used to sample over disorder (rather than studying specific realizations)
by maintaining a population of cavity messages and iteratively updating them.

Source files:
- Pop_dyn_Q-ISING_GAUSS-POPGEN.ipynb (Cells 10-12)
"""

import numpy as np
from .quantum_operators import make_expH, make_sx0, make_sz0, full_mess


def message_update(population, beta, K, sigma, G, sz0, sx0, message_function):
    """
    Update a cavity message based on the population of messages and couplings.
    
    This function randomly selects K messages from the population, generates
    random couplings, and computes a new cavity message using the quantum
    cavity method.
    
    Parameters:
    -----------
    population : np.ndarray
        The 2D array (Nx2) of all (h,b) messages in the population
    beta : float
        Inverse temperature
    K : int
        Coordination number (number of neighbors)
    sigma : float
        Standard deviation for coupling generation (will be rescaled by sqrt(K))
    G : float
        Transverse field strength
    sz0 : np.ndarray
        sz operator for the cavity site (precomputed for efficiency)
    sx0 : np.ndarray
        sx operator for the cavity site (precomputed for efficiency)
    message_function : callable
        Function that computes messages from expectation values (typically full_mess)
    
    Returns:
    --------
    tuple (float, float)
        Updated cavity messages (new_h, new_b)
    
    Notes:
    ------
    The coupling generation rescales sigma by sqrt(K) to remove K dependence
    of the critical temperature in the simplest mean-field model.
    
    Source:
    -------
    Pop_dyn_Q-ISING_GAUSS-POPGEN.ipynb, Cell 10
    """
    # Step 1: Randomly select messages corresponding to the couplings
    size_pop = np.shape(population)[0]
    rand_indx = np.random.randint(0, high=size_pop, size=K)
    
    selected_messages = population[rand_indx, :]
    
    # Generate random couplings (rescaled by sqrt(K))
    J_arr = np.random.normal(loc=0.0, scale=sigma/np.sqrt(K), size=K)
    # Note: rescaling sigma with K removes K dependence on critical line
    
    # Step 2: Compute matrix exponential
    expmat = make_expH(beta, K, J_arr, G, selected_messages[:, 0], selected_messages[:, 1])
    
    # Step 3: Get new message from cavity magnetizations
    new_h, new_b = message_function(beta, sz0, sx0, expmat)
    
    return new_h, new_b


def population_dynamics(
    N, max_iterations, window_size, K, sigma, G, beta, message_update_func, message_func
):
    """
    Perform population dynamics with convergence criterion based on median and IQR stability.
    
    This function maintains a population of N cavity messages and iteratively
    updates them. Convergence is checked based on the stability of the median
    and interquartile range (IQR) of both h and b populations.
    
    Parameters:
    -----------
    N : int
        Size of the population of messages
    max_iterations : int
        Maximum number of iterations allowed
    window_size : int
        Number of iterations to consider for the convergence check
    K : int
        Coordination number
    sigma : float
        Standard deviation for coupling generation
    G : float
        Transverse field strength
    beta : float
        Inverse temperature
    message_update_func : callable
        Function to update a cavity message (typically message_update)
    message_func : callable
        Function representing the message computation (typically full_mess)
    
    Returns:
    --------
    tuple
        (pop_h, pop_b, medians_h, medians_b, iqrs_h, iqrs_b, means_h, means_b, stds_h, stds_b)
        - pop_h, pop_b: Final converged populations
        - medians_h, medians_b: Time series of medians
        - iqrs_h, iqrs_b: Time series of IQRs
        - means_h, means_b: Time series of means
        - stds_h, stds_b: Time series of standard deviations
    
    Source:
    -------
    Pop_dyn_Q-ISING_GAUSS-POPGEN.ipynb, Cell 12
    """
    # Initialize the population
    # h fields: small random values around zero
    # b fields: small random values offset by G*K/(K+1) (mean-field expectation)
    population = np.empty([N, 2])
    population[:, 0] = np.random.uniform(-0.01, 0.01, N)
    population[:, 1] = np.random.uniform(-0.01, 0.01, N) + G * K / (K + 1)

    # Lists to track medians and IQRs (used for convergence)
    medians_h = []
    medians_b = []
    iqrs_h = []
    iqrs_b = []

    # Lists to track means and stds (for analysis, not convergence)
    means_h = []
    stds_h = []
    means_b = []
    stds_b = []

    # Track changes in median and IQR over iterations
    changes_median_h = []
    changes_iqr_h = []
    changes_median_b = []
    changes_iqr_b = []

    t = 0
    converged = False

    # Precompute operators (they don't depend on the iteration)
    sx0 = make_sx0(K)
    sz0 = make_sz0(K)

    while t < max_iterations and not converged:
        # Select a random index in the population to update
        index = np.random.randint(0, N)
    
        # Update the h and b field for this index
        new_h, new_b = message_update_func(
            population, beta, K, sigma, G, sz0, sx0, message_func
        )
        population[index] = [new_h, new_b]

        # Calculate the current median and IQR
        current_median_h = np.median(population[:, 0])
        current_iqr_h = np.quantile(population[:, 0], 0.75) - np.quantile(population[:, 0], 0.25)
        
        current_median_b = np.median(population[:, 1])
        current_iqr_b = np.quantile(population[:, 1], 0.75) - np.quantile(population[:, 1], 0.25)

        # Calculate the current mean and std
        current_mean_h = np.mean(population[:, 0])
        current_std_h = np.std(population[:, 0])
        
        current_mean_b = np.mean(population[:, 1])
        current_std_b = np.std(population[:, 1])

        # Track changes after the first iteration
        if t > 0:
            changes_median_h.append(abs(current_median_h - medians_h[-1]))
            changes_iqr_h.append(abs(current_iqr_h - iqrs_h[-1]))
            
            changes_median_b.append(abs(current_median_b - medians_b[-1]))
            changes_iqr_b.append(abs(current_iqr_b - iqrs_b[-1]))

        # Store the median and IQR
        medians_h.append(current_median_h)
        iqrs_h.append(current_iqr_h)

        medians_b.append(current_median_b)
        iqrs_b.append(current_iqr_b)

        # Store the mean and std
        means_h.append(current_mean_h)
        stds_h.append(current_std_h)

        means_b.append(current_mean_b)
        stds_b.append(current_std_b)

        # Adaptive convergence thresholds
        # Early iterations: use fixed small threshold
        # Later iterations: use fraction of current IQR
        if t < 100 * N:
            epsilon = 10**(-5)
            epsilon_median_h = epsilon / N
            epsilon_iqr_h = epsilon / np.sqrt(N)
            epsilon_median_b = epsilon / N
            epsilon_iqr_b = epsilon / np.sqrt(N)
        else:
            epsilon_h = current_iqr_h * 0.01
            epsilon_b = current_iqr_b * 0.01
            epsilon_median_h = epsilon_h / N
            epsilon_iqr_h = epsilon_h / np.sqrt(N)
            epsilon_median_b = epsilon_b / N
            epsilon_iqr_b = epsilon_b / np.sqrt(N)
        
        # Check convergence if enough iterations have passed
        if t >= N:
            # Check if changes in median and IQR are below threshold
            median_h_converged = np.mean(changes_median_h[-window_size:]) < epsilon_median_h
            iqr_h_converged = np.mean(changes_iqr_h[-window_size:]) < epsilon_iqr_h

            median_b_converged = np.mean(changes_median_b[-window_size:]) < epsilon_median_b
            iqr_b_converged = np.mean(changes_iqr_b[-window_size:]) < epsilon_iqr_b

            # Special cases: if IQR or median are extremely small, consider converged
            if current_iqr_h < 10**(-5):
                iqr_h_converged = True
                iqr_b_converged = True

            if current_median_h < 10**(-5):
                median_h_converged = True
                median_b_converged = True

            # All criteria must be satisfied
            if median_h_converged and iqr_h_converged and median_b_converged and iqr_b_converged:
                print(f"Convergence achieved after {t + 1} iterations.")
                converged = True

        # Increment iteration counter
        t += 1

        # Periodically print progress
        if (t + 1) % int(max_iterations / 10) == 0:
            print(f"Iteration {t + 1}/{max_iterations}...")

    if not converged:
        print("Reached maximum iterations without full convergence.")

    return (
        population[:, 0], population[:, 1],
        np.array(medians_h), np.array(medians_b),
        np.array(iqrs_h), np.array(iqrs_b),
        np.array(means_h), np.array(means_b),
        np.array(stds_h), np.array(stds_b)
    )


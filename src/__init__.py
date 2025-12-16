"""
Quantum Spin Glass Repository

This package implements the cavity method (belief propagation) and population
dynamics for quantum spin glasses, specifically the transverse field quantum
Ising model with Gaussian disorder.
"""

from .quantum_operators import (
    make_betaHeff,
    make_expH,
    make_sx0,
    make_sz0,
    expect_val,
    full_mess
)

from .graph_utils import (
    generate_random_regular_graph,
    initialize_couplings,
    couplings_dict_to_matrix
)

from .cavity_method import (
    initialize_messages,
    F_func,
    belief_propagation,
    compute_susceptibility_prop
)

from .population_dynamics import (
    message_update,
    population_dynamics
)

from .analysis_utils import (
    compute_from_population_dynamics,
    extract_parameters_from_pkl,
    convert_results_to_arrays,
    mean_excess_function,
    get_pop_h,
    get_pop_b,
    get_stds_h,
    get_stds_b,
    get_means_h,
    get_means_b,
    get_iqrs_h,
    get_iqrs_b,
    get_medians_h,
    get_medians_b,
    gen_ipr,
    setup_plotting,
    create_figure,
    save_figure,
    invert_function,
    find_near_critical_Gamma,
    compute_ipr_and_gamma_for_config,
    run_experiments
)


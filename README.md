# Quantum Spin Glass Repository

This repository contains a condensed implementation of the cavity method (belief propagation) and population dynamics for studying quantum spin glasses, specifically the transverse field quantum Ising model with Gaussian disorder. 

References [1,2] have been used to develop this code.

This code has been used to develop certain results in the last chapter of my PhD thesis, which is available [\here](https://doi.org/10.3929/ethz-c-000790113).

The condensation of the multiple codes used for this project, and the refinement of the repository, have been partially carried out using Cursor.

## Overview

This code implements two main approaches:

1. **Cavity Method (Belief Propagation)**: Iteratively solves the cavity equations on a specific graph realization to find the fixed-point cavity messages.

2. **Population Dynamics**: Samples over disorder by maintaining a population of cavity messages and iteratively updating them, allowing study of the distribution of cavity fields rather than specific realizations.


The code can be used directly by importing from the `src` module. See the example notebooks for usage.

## Usage


The cavity method solves the quantum Ising model on a specific graph realization. Population dynamics samples over disorder (multiple realizations). Check the example notebooks to see how to use the relevant functions.


### Example Notebooks

See the `examples/` directory for detailed Jupyter notebooks demonstrating:
- How to use the cavity method
- How to use population dynamics
- How to analyze results


## Physical Model

The code implements the transverse field quantum Ising model:

**Hamiltonian**: H = -Σ_{ij} J_{ij} σ^z_i σ^z_j - Γ Σ_i σ^x_i

where:
- J_{ij} are Gaussian random couplings
- Γ is the transverse field strength
- σ^x, σ^z are Pauli matrices

The cavity method solves this on random regular graphs, where each spin has exactly `degree` neighbors.

## Convergence Criteria

### Belief Propagation

- **Strict convergence**: Maximum message change < tolerance
- **RSB detection**: No decrease in error after sufficient iterations

### Population Dynamics

- **Median convergence**: Change in median < threshold
- **IQR convergence**: Change in IQR < threshold
- **Adaptive thresholds**: Early iterations use fixed thresholds, later iterations use fraction of current IQR

The reason to use median and IQR rather than mean and standard deviation is that the distributions may be heavy-tailed.

## Saving Results

Population dynamics results can be saved for later analysis:

```python
import pickle

results = {
    "parameters": {
        "N": N,
        "sigma": sigma,
        "K": K,
        "beta": beta,
        "G": G
    },
    "pop_h": final_population_h,
    "pop_b": final_population_b,
    "medians_h": medians_h,
    "medians_b": medians_b,
    "iqrs_h": iqrs_h,
    "iqrs_b": iqrs_b,
    "means_h": means_h,
    "means_b": means_b,
    "stds_h": stds_h,
    "stds_b": stds_b
}

with open("results.pkl", "wb") as f:
    pickle.dump(results, f)
```

## References

This code implements the cavity method (also known as belief propagation) for quantum spin glasses. Specifically, the operatorial cavity method, for spin-1/2 and restricting the message density matrices to single spins (4x4).

- [1] Bilgin, E. & Poulin, D. Coarse-grained belief propagation for simulation of interacting quantum systems at all temperatures. Phys. Rev. B 81, 054106 (5 2010).
- [2] Dimitrova, O. & Mézard, M. The cavity method for quantum disordered systems: from transverse random field ferromagnets to directed polymers in random media. Journal of Statistical Mechanics: Theory and Experiment 2011, P01020 (2011).

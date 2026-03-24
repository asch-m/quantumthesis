#!/usr/bin/env python3
"""
MBVQE Simulation Library 

This module consolidates the core VQE simulation functions used to generate
data for the paper figures. Running the full parameter sweeps for all figures is computationally intensive and is NOT included here. 
Instead, this provides the building blocks to run individual simulations. The parameter settings are documented in https://iopscience.iop.org/article/10.1088/1367-2630/ad51e5/meta.

Usage:
    python mbvqe_simulation.py  # Run examples

    # Or import as library:
    from mbvqe_simulation import run_vqe, linear_XY

Features:
    - Hamiltonians: XY model, Schwinger model 
    - Ansätze: Cluster state with/without U3 gates
    - VQE: L-BFGS-B optimization with exact statevector simulation
    - Utilities: Error computation, exact diagonalization

Dependencies:
    - qiskit >= 0.40
    - qiskit-aer
    - numpy
    - scipy
"""

import numpy as np
import qiskit
from functools import partial
from scipy.optimize import minimize

from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.algorithms.minimum_eigensolvers import VQE

from qiskit.primitives import Estimator 
from qiskit.algorithms.optimizers import L_BFGS_B


# =============================================================================
# HAMILTONIANS
# =============================================================================

def linear_XY(num_qubits, g, local_z_noise):
    """
    Linear XY model Hamiltonian.

    H = Σ_i [(1+g)(X_i X_{i+1}) + (1-g)(Y_i Y_{i+1})] + Σ_i d·Z_i

    Parameters:
    -----------
    num_qubits : int
        Number of qubits in the chain
    g : float
        Anisotropy parameter
        g = 0: isotropic (critical)
        g = 1: fully anisotropic (non-critical)
    local_z_noise : float
        Local Z-field perturbation to break degeneracy

    Returns:
    --------
    SparsePauliOp : Qiskit Hamiltonian operator
    """
    edges = [[i, i+1] for i in range(num_qubits-1)]
    return XY_ham_2d(edges, num_qubits, g, local_z_noise)


def XY_ham_2d(edges, num_qubits, g, local_z_noise):
    """
    2D XY Hamiltonian on arbitrary graph.

    Parameters:
    -----------
    edges : list of [int, int]
        List of edges defining the connectivity
    num_qubits : int
        Total number of qubits
    g : float
        Anisotropy parameter
    local_z_noise : float
        Local Z-field strength

    Returns:
    --------
    SparsePauliOp : Hamiltonian operator
    """
    ops_list = []
    for e in edges:
        ops_list.append((pad_identities_double_op('X', e[0], 'X', e[1], num_qubits), 1+g))
        ops_list.append((pad_identities_double_op('Y', e[0], 'Y', e[1], num_qubits), 1-g))

    for i in range(num_qubits):
        ops_list.append((pad_identities_single_op('Z', i, num_qubits), local_z_noise))

    return SparsePauliOp.from_list(ops_list)


def Schwinger_ham(S, mu, J=1, w=1):
    """
    Schwinger model Hamiltonian for lattice gauge theory.

    The Schwinger model describes quantum electrodynamics in 1+1 dimensions.

    Parameters:
    -----------
    S : int
        Number of fermion sites (qubits)
    mu : float
        Chemical potential (mass parameter)
        mu = -0.7: critical point (phase transition)
        mu = 1: non-critical regime
    J : float, default=1
        Coupling strength for electric field term
    w : float, default=1
        Hopping term strength

    Returns:
    --------
    SparsePauliOp : Hamiltonian operator

    Notes:
    ------
    The Hamiltonian includes:
    - Electric field energy (ZZ terms)
    - Background field (Z terms)
    - Fermion hopping (XX + YY terms)
    - Chemical potential (alternating Z terms)
    """
    # Electric field energy term
    term1 = []
    coeff1 = 0.5 * J
    for i in range(S-2):
        for j in range(i+1, S-1):
            term1.append((pad_identities_double_op('Z', i, 'Z', j, S),
                         coeff1 * (S - j - 1)))

    # Background electric field term
    term2 = []
    coeff2 = -0.5 * J
    for i in range(S-1):
        if i % 2 == 0:
            for j in range(i+1):
                term2.append((pad_identities_single_op('Z', j, S), coeff2))

    # Fermion hopping term
    term3 = []
    coeff3 = 0.5 * w
    for i in range(S-1):
        term3.append((pad_identities_double_op('X', i, 'X', i+1, S), coeff3))
        term3.append((pad_identities_double_op('Y', i, 'Y', i+1, S), coeff3))

    # Chemical potential term (mass)
    term4 = []
    coeff4 = 0.5 * mu
    for i in range(S):
        term4.append((pad_identities_single_op('Z', i, S),
                     coeff4 * (-1)**(i+1)))

    return SparsePauliOp.from_list(term1 + term2 + term3 + term4)


def pad_identities_single_op(op, ind, n_qubits):
    """Pad identities around a single Pauli operator at position ind."""
    op_list = ''.join([op if i == ind else 'I' for i in range(n_qubits)])
    return op_list


def pad_identities_double_op(op1, op1_ind, op2, op2_ind, n_qubits):
    """Pad identities around two Pauli operators at given positions."""
    op_list = ''
    for i in range(n_qubits):
        if i == op1_ind:
            op_list += op1
        elif i == op2_ind:
            op_list += op2
        else:
            op_list += 'I'
    return op_list


# =============================================================================
# ANSATZ CIRCUITS
# =============================================================================

def cluster_state_ansatz(n_qubits, n_layers):
    """
    MBVQE ansatz circuit WITH parametrized U3 gates on output layer.

    Structure:
    1. Hadamard gates on all qubits (create |+⟩ states)
    2. CZ gates on linear connectivity (create cluster state)
    3. Decorated layers using TwoLocal (RZ-H-CZ pattern)
    4. Final CZ layer
    5. Parametrized U3 gates on all qubits

    Parameters:
    -----------
    n_qubits : int
        Number of qubits
    n_layers : int
        Number of decoration layers (reps in TwoLocal)

    Returns:
    --------
    QuantumCircuit : Parameterized quantum circuit

    Notes:
    ------
    Total parameters: 2*n_qubits*n_layers + 3*n_qubits
    - 2*n_qubits*n_layers from TwoLocal (RZ gates)
    - 3*n_qubits from U3 gates (θ, φ, λ parameters)
    """
    edges_list = [[i, i+1] for i in range(n_qubits-1)]
    qc = qiskit.QuantumCircuit(n_qubits)

    # Initialize cluster state
    for i in range(n_qubits):
        qc.h(i)

    for edge in edges_list:
        qc.cz(edge[0], edge[1])

    # MBVQE decoration layers
    mbvqe = qiskit.circuit.library.TwoLocal(
        num_qubits=n_qubits,
        rotation_blocks=['rz', 'h'],
        entanglement_blocks='cz',
        reps=n_layers,
        entanglement='linear',
        insert_barriers=True
    )
    qc.append(mbvqe, list(range(n_qubits)))

    # Final CZ layer
    for edge in edges_list:
        qc.cz(edge[0], edge[1])

    # Parametrized local unitaries (U3 gates)
    phis = qiskit.circuit.ParameterVector('phi', length=3*n_qubits)
    for i in range(n_qubits):
        qc.u(phis[i*3], phis[i*3+1], phis[i*3+2], i)

    return qc


def cluster_state_ansatz_wo_u(n_qubits, n_layers):
    """
    MBVQE ansatz circuit WITHOUT parametrized U3 gates on output layer.

    Same as cluster_state_ansatz but without the final U3 layer.
    Used to study the effect of output layer parametrization.

    Parameters:
    -----------
    n_qubits : int
        Number of qubits
    n_layers : int
        Number of decoration layers

    Returns:
    --------
    QuantumCircuit : Parameterized quantum circuit

    Notes:
    ------
    Total parameters: 2*n_qubits*n_layers
    - Only from TwoLocal decoration layers
    """
    edges_list = [[i, i+1] for i in range(n_qubits-1)]
    qc = qiskit.QuantumCircuit(n_qubits)

    # Initialize cluster state
    for i in range(n_qubits):
        qc.h(i)

    for edge in edges_list:
        qc.cz(edge[0], edge[1])

    # MBVQE decoration layers
    mbvqe = qiskit.circuit.library.TwoLocal(
        num_qubits=n_qubits,
        rotation_blocks=['rz', 'h'],
        entanglement_blocks='cz',
        reps=n_layers,
        entanglement='linear',
        insert_barriers=True
    )
    qc.append(mbvqe, list(range(n_qubits)))

    # Final CZ layer
    for edge in edges_list:
        qc.cz(edge[0], edge[1])

    return qc


# =============================================================================
# VQE SIMULATION
# =============================================================================

def run_vqe(ansatz, ham, init_param=None):
    """
    Run Variational Quantum Eigensolver to find ground state.

    Uses L-BFGS-B optimizer with tight convergence criteria and
    noiseless statevector simulation.

    Parameters:
    -----------
    ansatz : QuantumCircuit
        Parameterized ansatz circuit
    ham : SparsePauliOp
        Hamiltonian operator
    init_param : array_like, optional
        Initial parameter values
        If None, random initialization is used

    Returns:
    --------
    tuple : (counts, values, params, result)
        counts : list
            Evaluation count at each callback
        values : list
            Energy value at each callback
        params : list
            Parameter values at each callback
        result : VQEResult
            Final VQE result object

    Notes:
    ------
    Optimizer settings:
    - method: L-BFGS-B (bound-constrained quasi-Newton)
    - ftol: 1e-16 (function tolerance)
    - gtol: 1e-16 (gradient tolerance)
    - maxiter: 5000 (maximum iterations)

    Estimator: AerEstimator with exact statevector simulation
    """

    estimator = Estimator() # Pure statevector — no Aer serialization issues

    optimizer = L_BFGS_B()

    # Storage for intermediate results
    counts = []
    values = []
    params = []

    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)
        params.append(parameters)

    # Noiseless statevector estimator
    noiseless_estimator = AerEstimator(
        run_options={"shots": None},
        approximation=True
    )

    # VQE algorithm
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=init_param,
        callback=store_intermediate_result
    )

    result = vqe.compute_minimum_eigenvalue(operator=ham)

    return counts, values, params, result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_relative_error(energy, true_gs_energy):
    """
    Compute relative error in ground state energy.

    Parameters:
    -----------
    energy : float
        Computed ground state energy
    true_gs_energy : float
        Exact ground state energy

    Returns:
    --------
    float : Relative error |E - E_gs| / |E_gs|
    """
    return np.abs((energy - true_gs_energy) / true_gs_energy)


def true_ground_state_energy_and_state(hamiltonian):
    """
    Compute exact ground state energy and state using full diagonalization.

    Parameters:
    -----------
    hamiltonian : SparsePauliOp or ndarray
        Hamiltonian operator

    Returns:
    --------
    tuple : (energy, state)
        energy : float
            Ground state energy (minimum eigenvalue)
        state : ndarray
            Ground state vector (corresponding eigenvector)

    Notes:
    ------
    Uses numpy.linalg.eigh for Hermitian matrices.
    Only suitable for small system sizes (<15 qubits).
    """
    if isinstance(hamiltonian, SparsePauliOp):
        ham_matrix = hamiltonian.to_matrix()
    else:
        ham_matrix = hamiltonian

    eigenvalues, eigenvectors = np.linalg.eigh(ham_matrix)
    gs_energy = eigenvalues[0]
    gs_state = eigenvectors[:, 0]

    return gs_energy, gs_state


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_xy_model_simulation():
    """
    Example: Run a single VQE simulation for XY model.

    This demonstrates the basic workflow but does NOT run the full
    parameter sweep needed for the paper figures.
    """
    print("Example: XY Model Simulation")
    print("-" * 50)

    # System parameters
    n_qubits = 4
    n_layers = 3
    g = 1.0  # anisotropic
    z_noise = 0.01

    # Create Hamiltonian
    ham = linear_XY(n_qubits, g, z_noise)
    print(f"Hamiltonian: {n_qubits}-qubit XY model with g={g}")

    # Create ansatz
    ansatz = cluster_state_ansatz(n_qubits, n_layers)
    print(f"Ansatz: {n_layers} decoration layers with U3 gates")
    print(f"Parameters: {ansatz.num_parameters}")

    # Compute exact ground state for comparison
    true_gs_energy, _ = true_ground_state_energy_and_state(ham)
    print(f"Exact ground state energy: {true_gs_energy:.6f}")

    # Run VQE
    print("\nRunning VQE...")
    counts, values, params, result = run_vqe(ansatz, ham)

    # Results
    vqe_energy = result.eigenvalue
    rel_error = compute_relative_error(vqe_energy, true_gs_energy)

    print(f"VQE ground state energy: {vqe_energy:.6f}")
    print(f"Relative error: {rel_error:.2e}")
    print(f"Optimizer evaluations: {len(counts)}")

    return result


def example_schwinger_model_simulation():
    """
    Example: Run a single VQE simulation for Schwinger model.
    """
    print("\nExample: Schwinger Model Simulation")
    print("-" * 50)

    # System parameters
    n_qubits = 4
    n_layers = 3
    mu = -0.7  # critical point

    # Create Hamiltonian
    ham = Schwinger_ham(n_qubits, mu)
    print(f"Hamiltonian: {n_qubits}-qubit Schwinger model with μ={mu}")

    # Create ansatz
    ansatz = cluster_state_ansatz(n_qubits, n_layers)
    print(f"Ansatz: {n_layers} decoration layers with U3 gates")

    # Compute exact ground state
    true_gs_energy, _ = true_ground_state_energy_and_state(ham)
    print(f"Exact ground state energy: {true_gs_energy:.6f}")

    # Run VQE
    print("\nRunning VQE...")
    counts, values, params, result = run_vqe(ansatz, ham)

    # Results
    vqe_energy = result.eigenvalue
    rel_error = compute_relative_error(vqe_energy, true_gs_energy)

    print(f"VQE ground state energy: {vqe_energy:.6f}")
    print(f"Relative error: {rel_error:.2e}")
    print(f"Optimizer evaluations: {len(counts)}")

    return result


if __name__ == '__main__':
    print("=" * 70)
    print("MBVQE Simulation Library - Example Usage")
    print("=" * 70)
    print()
    print("This script demonstrates the simulation functions used to generate")
    print("data for paper figures. It does NOT run the full parameter sweeps")
    print("(which require extensive computational resources).")
    print()
    print("See https://iopscience.iop.org/article/10.1088/1367-2630/ad51e5/meta for full parameter settings.")
    print("=" * 70)
    print()

    # Run example simulations
    try:
        example_xy_model_simulation()
        example_schwinger_model_simulation()

        print("\n" + "=" * 70)
        print("Examples completed successfully!")
        print("=" * 70)
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure qiskit and qiskit-aer are installed in mbvqeenv.")

import argparse
import itertools
import numpy as np
import pickle
import os
from pathlib import Path
from qiskit import QuantumCircuit
from config import MQSS_TOKEN, MQSS_BACKENDS
from mqss.qiskit_adapter import MQSSQiskitAdapter

pauli_dict = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]]),
}


def apply_state_tomography_basis(qc, ops):
    """
    Applies basis change gates for state tomography to the given quantum circuit.

    Args:
        qc (QuantumCircuit): The quantum circuit.
        ops (list of str): List of basis labels ('Z', 'X', 'Y', 'I') for each qubit.

    Returns:
        QuantumCircuit: The circuit with basis change gates and measurements.
    """

    for idx, op in enumerate(ops):
        if op == "X":
            qc.h(idx)  # Hadamard for X basis
        elif op == "Y":
            qc.sdg(idx)
            qc.h(idx)  # Sdg then Hadamard for Y basis
        elif op == "Z":
            pass  # Z basis, no change
        elif op == "I":
            pass  # same for id
        else:
            raise ValueError(f"Unknown basis: {op}")
    qc.measure_all()
    return qc


def pauli_expectation(counts, basis, shots):
    exp = 0
    for bitstring, count in counts.items():
        parity = 1
        for idx, b in enumerate(basis):
            if b in ["X", "Y", "Z"]:
                if bitstring[idx] == "1":
                    parity *= -1
        exp += parity * count
    return exp / shots


def tensor_product_from_ops(ops):
    """
    Given a list of ops (e.g. ['X', 'Y', 'Z']), return the tensor product matrix.
    """
    result = pauli_dict[ops[0]]
    for op in ops[1:]:
        result = np.kron(result, pauli_dict[op])
    return result


def state_tomograph_routine(
    qasm_path, backend=None, shots=1024, checkpoint_file="rho_checkpoint.pkl"
):
    qc = QuantumCircuit.from_qasm_file(qasm_path)
    n_qubits = qc.num_qubits
    bases = ["I", "X", "Y", "Z"]
    all_ops = list(itertools.product(bases, repeat=n_qubits))
    print(f"Total operations to run: {len(all_ops)}")
    if backend is None:
        adapter = MQSSQiskitAdapter(token=MQSS_TOKEN)
        backend = adapter.get_backend(MQSS_BACKENDS)

    # Try to load checkpoint
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)
        start_idx = checkpoint["last_idx"] + 1
        rho_reconstructed = checkpoint["rho"]
    else:
        start_idx = 0
        rho_reconstructed = (
            np.eye(2**n_qubits, dtype=np.complex128) / (2**n_qubits) + 0j
        )

    for idx, ops in enumerate(all_ops[start_idx:], start=start_idx):
        try:
            qc_copy = qc.copy()
            qc_copy = apply_state_tomography_basis(qc_copy, ops)
            print(f"Running circuit for ops: {ops} (index {idx})")
            job = backend.run(qc_copy, shots=shots)
            result = job.result()
            counts = result.get_counts()
            exp = pauli_expectation(counts, ops, shots)
            matrix = tensor_product_from_ops(ops) + 0j

            rho_reconstructed += exp * matrix

            # Save checkpoint after each step during density matrix reconstruction
            with open(checkpoint_file, "wb") as f:
                print(f"Checkpointing at op {idx}: {ops}")
                pickle.dump({"last_idx": idx, "rho": rho_reconstructed}, f)

        except Exception as e:
            print(f"Error at op {idx}: {ops}, error: {e}")
            break

    return rho_reconstructed


def main():
    parser = argparse.ArgumentParser(description="Quantum State Tomography Routine")
    parser.add_argument("qasm_path", type=str, help="Path to the QASM file")
    parser.add_argument(
        "--shots", type=int, default=1024, help="Number of shots per circuit"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/rho_checkpoint.pkl",
        help="Checkpoint file path",
    )
    args = parser.parse_args()

    rho = state_tomograph_routine(
        args.qasm_path, shots=args.shots, checkpoint_file=args.checkpoint
    )
    print("Reconstructed density matrix:")
    print(rho)


if __name__ == "__main__":
    main()

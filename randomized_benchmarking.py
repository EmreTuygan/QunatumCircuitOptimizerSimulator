import argparse
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford
from qiskit import transpile, qasm3


def generate_rb_circuit(n_qubits, n_layers, seed=None):
    """
    Generate a randomized benchmarking circuit using Clifford gates.

    Args:
        n_qubits (int): Number of qubits.
        n_layers (int): Number of Clifford layers.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        QuantumCircuit: The randomized benchmarking circuit.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    cliffords = []

    for _ in range(n_layers):
        # Generate a random Clifford for all qubits
        cliff = random_clifford(n_qubits, seed=int(rng.integers(1e9)))
        qc.append(cliff.to_circuit(), range(n_qubits))
        cliffords.append(cliff)

    # Compute the inverse Clifford to return to |0...0>
    total_clifford = cliffords[0]
    for c in cliffords[1:]:
        total_clifford = total_clifford.compose(c)
    inverse_clifford = total_clifford.adjoint()
    qc.append(inverse_clifford.to_circuit(), range(n_qubits))

    qc.measure_all()
    return qc


def main():
    parser = argparse.ArgumentParser(
        description="Randomized Benchmarking Circuit Generator"
    )
    parser.add_argument("--qubits", type=int, required=True, help="Number of qubits")
    parser.add_argument(
        "--layers", type=int, required=True, help="Number of Clifford layers"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--output", type=str, default=None, help="Output QASM file (optional)"
    )
    args = parser.parse_args()

    qc = generate_rb_circuit(args.qubits, args.layers, args.seed)

    basis_gates = ["cx", "cz", "rx", "rz", "ry", "h"]
    qc = transpile(qc, basis_gates=basis_gates, optimization_level=1)
    print("Transpiled circuit to basis gates:", basis_gates)

    if args.output:
        # qc.qasm(filename=args.output)

        with open(args.output, "w") as f:
            qasm3.dump(qc, f)

        print(f"QASM saved to {args.output}")
    else:
        output_qasm = f"data/rb_circuit_{args.qubits}_{args.layers}_{args.seed}.qasm"
        with open(args.output, "w") as f:
            qasm3.dump(qc, f)

        print(f"QASM saved to {args.output}")


if __name__ == "__main__":
    main()

from qiskit.circuit.library import CPhaseGate, RGate
from qiskit.transpiler import Target
from qiskit.circuit import Parameter
from qiskit.circuit.library import CPhaseGate


def get_qiskit_machine_model() -> Target:

    edges = [
        (17, 18), (18, 19),
        (12, 13), (13, 14), (14, 15), (15, 16),
        (7, 8), (8, 9), (9, 10), (10, 11),
        (2, 3), (3, 4), (4, 5), (5, 6),
        (0, 1),
        
        (13, 17), (14, 18), (15, 19),
        (16, 11), (15, 10), (14, 9), (13, 8), (12, 7),
        (11, 6), (10, 5), (9, 4), (8, 3), (7, 2),
        (4, 1), (3, 0)
    ]

    # Create the Target object with correct number of qubits
    target = Target(num_qubits=20)

    # Add CPhaseGate to all bidirectional edges
    cphase_props = {}
    for q0, q1 in edges:
        cphase_props[(q0, q1)] = None
        cphase_props[(q1, q0)] = None
    theta = Parameter("theta")
    target.add_instruction(CPhaseGate(theta), cphase_props)

    # Add RGate to each individual qubit (single-qubit instruction)
    theta2 = Parameter("theta")
    phi = Parameter("phi")
    # Create the full qubit set
    rgate_qubits = { (q,): None for q in range(20) }

    # Now add the instruction just once for all qubits
    target.add_instruction(RGate(theta2, phi), rgate_qubits)

    return target


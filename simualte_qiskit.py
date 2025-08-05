from qiskit import QuantumCircuit
from pathlib import Path
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit_aer.noise import NoiseModel
from qiskit.transpiler import Target

from topology import get_qiskit_machine_model

model = get_qiskit_machine_model()

# This should be crefactored to include our noise model
def build_noise_model(target: Target) -> NoiseModel:
    noise_model = NoiseModel()
    return noise_model


# 1. Load QASM circuit
circ = "adder4_iteration1.qasm"
qasm_path = Path(f"qasm_optimized_Qiskit/adder/opt0/{circ}")
qc = QuantumCircuit.from_qasm_file(qasm_path)

# add meaurments
observable = SparsePauliOp("Z" * qc.num_qubits)
params = [0.1] * qc.num_parameters
print(qc.parameters)


# Exact (noiseless)
print("Running Estimator (noiseless)...")
exact_estimator = Estimator()
job = exact_estimator.run([(qc, observable, params)])
exact_value = float(job.result()[0].data.evs)
print(f"Noiseless expectation value: {exact_value}")

# Noisy
print("Running Estimator (with noise)...")
noise_model = build_noise_model(get_qiskit_machine_model())
noisy_estimator = Estimator(
    options={"backend_options": {"noise_model": noise_model}}
)
job = noisy_estimator.run([(qc, observable, params)])
noisy_value = float(job.result()[0].data.evs)
print(f"  Noisy expectation value: {noisy_value}")

# Sampler
measured_circuit = qc.copy()
measured_circuit.measure_all()
 
print("Running Sampler (with noise, 1000 shots)...")
noisy_sampler = Sampler(
    options=dict(backend_options=dict(noise_model=noise_model))
)
# The circuit needs to be transpiled to the AerSimulator target
job = noisy_sampler.run([(measured_circuit, params, 1000)])
counts = job.result()[0].data.meas.get_counts()
print("Sampled measurement counts:")
print(counts)
import networkx as nx
#import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
#from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import depolarizing_error
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_ibm_runtime.fake_provider import FakeVigo
#from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA

# -------------------------------
# 1. MaxCut graph (larger & denser)
# -------------------------------
#graph = nx.complete_graph(6)  # fully connected 6-node graph
graph = nx.gnp_random_graph(11, 0.9)
#pos = nx.spring_layout(graph, seed=42)
#nx.draw(graph, with_labels=True, pos=pos)
#plt.title("MaxCut: Complete Graph (6 nodes)")
#plt.show()

# -------------------------------
# 2. Convert to QUBO
# -------------------------------
maxcut = Maxcut(graph)
problem = maxcut.to_quadratic_program()
converter = QuadraticProgramToQubo()
qubo = converter.convert(problem)

# -------------------------------
# 3. Optimizer
# -------------------------------
#optimizer = COBYLA(maxiter=10)
optimizer = SPSA(maxiter=5)
# -------------------------------
# 4. Noise models and samplers
# -------------------------------

# (A) Ideal: AerSampler with shots=None simulates statevector
ideal_sampler = AerSampler()
ideal_sampler.options.shots = None

# (B) Sampling noise only
sampling_noise_sampler = AerSampler()
sampling_noise_sampler.options.shots = 256

# (C) Depolarizing gate noise + sampling
noise_model = NoiseModel()
error_1 = depolarizing_error(0.2, 1)
error_2 = depolarizing_error(0.3, 2)
noise_model.add_all_qubit_quantum_error(error_1, ['x', 'y', 'z', 'h', 'ry', 'rz', 'rx'])
noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

noisy_sampler = AerSampler()
noisy_sampler.options.shots = 256
noisy_sampler.options.noise_model = noise_model

# (D) FakeVigo
fake_backend = FakeVigo()
fake_noise_model = NoiseModel.from_backend(fake_backend)

fake_sampler = AerSampler()
fake_sampler.options.shots = 256
fake_sampler.options.noise_model = fake_noise_model

# -------------------------------
# 5. Run QAOA
# -------------------------------
def run_qaoa(sampler, description=""):
    print(f"\n=== {description} ===")
    qaoa = QAOA(optimizer=optimizer, reps=4, sampler=sampler)
    solver = MinimumEigenOptimizer(qaoa)
    result = solver.solve(qubo)
    print("Optimal solution:", result.x)
    print("Objective value:", result.fval)

# Run all three QAOA setups
run_qaoa(ideal_sampler, description="(A) Ideal - AerSampler (shots=None)")
run_qaoa(sampling_noise_sampler, description="(B) Sampling Noise Only")
run_qaoa(noisy_sampler, description="(C) Depolarizing Noise + Sampling")
run_qaoa(fake_sampler, description="(D) FakeVigo")

# -------------------------------
# 6. BONUS: Bell state circuit example
# -------------------------------
def run_bell_example(sampler, label=""):
    print(f"\n--- Bell State using {label} ---")
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()

    result = sampler.run([bell]).result()
    for i, dist in enumerate(result.quasi_dists):
        print(f"Result {i}: {dict(dist)}")

# Run Bell state on all three samplers
run_bell_example(ideal_sampler, label="Ideal")
run_bell_example(sampling_noise_sampler, label="Sampling Noise Only")
run_bell_example(noisy_sampler, label="Depolarizing Noise + Sampling")
run_bell_example(fake_sampler, label="Fake Vigo")
#this cell contains all the imports needed by the pipeline
#to run it on the browser: jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10
import os
import json
import time

import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import statistics
import ast
import csv
import optuna

from scipy.stats import mannwhitneyu, shapiro

from qiskit import qpy
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_optimization.converters import QuadraticProgramToQubo
from matplotlib import cm

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict, Counter

bootqa_programs = ["gsdtsr","paintcontrol", "iofrol", "elevator", "elevator2"]
bootqa_programs_rep_values = {"gsdtsr":1,"paintcontrol":1,"iofrol":1, "elevator":1, "elevator2":1}
experiments = 10

def get_data(data_name):
    """Read the datasets"""
    if data_name == "elevator":
        data = pd.read_csv("datasets/quantum_sota_datasets/elevator.csv", dtype={"cost": int, "input_div": float})
    elif data_name == "elevator2":
        data = pd.read_csv("datasets/quantum_sota_datasets/elevator.csv", dtype={"cost": int, "pcount": int, "dist": int})
    else:
        data = pd.read_csv("datasets/quantum_sota_datasets/" + data_name + ".csv", dtype={"time": float, "rate": float})
        data = data[data['rate'] > 0]
    return data


from collections import defaultdict

bootqa_clusters = dict()

for bootqa_program in bootqa_programs:
    data = get_data(bootqa_program)

    # Total suite metrics
    if bootqa_program == "elevator" or bootqa_program == "elevator2":
        test_cases_costs = data["cost"].tolist()
    else:
        test_cases_costs = data["time"].tolist()

    if bootqa_program == "elevator":
        test_cases_effectiveness = data["input_div"].tolist()
        # print(f"Tot suite cost: {sum(test_cases_costs)}")
        # print(f"Tot suite input div: {sum(test_cases_effectiveness)}")
    elif bootqa_program == "elevator2":
        test_cases_pcount = data["pcount"].tolist()
        test_cases_dist = data["dist"].tolist()
        # print(f"Tot suite cost: {sum(test_cases_costs)}")
        # print(f"Tot suite pcount: {sum(test_cases_pcount)}")
        # print(f"Tot suite dist: {sum(test_cases_dist)}")
    else:
        test_cases_effectiveness = data["rate"].tolist()
        # print(f"Tot suite cost: {sum(test_cases_costs)}")
        # print(f"Tot suite rate: {sum(test_cases_effectiveness)}")

    # Normalize data
    if bootqa_program != "elevator2":
        cluster_data = np.column_stack((test_cases_costs, test_cases_effectiveness))
    else:
        cluster_data = np.column_stack((test_cases_costs, test_cases_pcount, test_cases_dist))

    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(cluster_data)

    if bootqa_program == "elevator" or bootqa_program == "elevator2":
        num_clusters = 800
    if bootqa_program == "gsdtsr":
        num_clusters = 60
    if bootqa_program == "iofrol":
        num_clusters = 324
    if bootqa_program == "paintcontrol":
        num_clusters = 16

    max_cluster_dim = 7

    # Step 2: Perform K-Means Clustering
    start = time.time()
    linkage_matrix = linkage(normalized_data, method='ward')
    clusters = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

    # Organize test cases by cluster
    clustered_data = defaultdict(list)
    for idx, cluster_id in enumerate(clusters):
        clustered_data[cluster_id].append(idx)

    # Process clusters to ensure none exceed max_cluster_dim
    new_cluster_id = max(clustered_data.keys()) + 1  # Start new IDs after existing ones
    to_add = []  # Collect new smaller clusters

    for cluster_id, elements in list(clustered_data.items()):  # Avoid modifying dict during iteration
        if len(elements) > max_cluster_dim:
            num_splits = -(-len(elements) // max_cluster_dim)  # Ceiling division to get the required number of splits
            split_size = -(-len(elements) // num_splits)  # Recalculate to distribute elements evenly

            # Split while keeping sizes balanced
            parts = [elements[i:i + split_size] for i in range(0, len(elements), split_size)]

            # Ensure all new clusters are within max_cluster_dim
            for part in parts:
                if len(part) > max_cluster_dim:
                    raise ValueError(
                        f"A split cluster still exceeds max_cluster_dim ({len(part)} > {max_cluster_dim})!")

            # Add new parts to the new clusters
            to_add.extend(parts)

            # Remove original large cluster
            del clustered_data[cluster_id]

    # Assign new IDs to split parts
    for part in to_add:
        if part:  # Only add if the part is non-empty
            clustered_data[new_cluster_id] = part
            new_cluster_id += 1
    end = time.time()
    # print("SelectQAOA Decomposition Time(ms): " + str((end-start)*1000))

    bootqa_clusters[bootqa_program] = clustered_data

    # Step 3: Calculate the metrics for each refined cluster
    cluster_metrics = {}
    for cluster_id, members in clustered_data.items():
        tot_cluster_costs = sum(test_cases_costs[i] for i in members)
        if bootqa_program != "elevator2":
            tot_cluster_effectiveness = sum(test_cases_effectiveness[i] for i in members)
        else:
            tot_cluster_pcount = sum(test_cases_pcount[i] for i in members)
            tot_cluster_dist = sum(test_cases_dist[i] for i in members)
        if bootqa_program != "elevator2":
            cluster_metrics[cluster_id] = {
                "tot_cluster_cost": tot_cluster_costs,
                "tot_cluster_rates": tot_cluster_effectiveness
            }
        else:
            cluster_metrics[cluster_id] = {
                "tot_cluster_cost": tot_cluster_costs,
                "tot_cluster_pcount": tot_cluster_pcount,
                "tot_cluster_dist": tot_cluster_dist
            }
        # print(f"Cluster {cluster_id + 1} metrics:")
        # print(f"Test Cases: {members}")
        # print(f" - Num. Test Cases: {len(members):.2f}")
        # print(f" - Execution Cost: {tot_cluster_costs:.2f}")
        if bootqa_program != "elevator2":
            print(f" - Failure Rate: {tot_cluster_effectiveness}")
        else:
            print(f" - PCount: {tot_cluster_pcount}")
            print(f" - Dist: {tot_cluster_dist}")

    print("===========================================================================")

    for cluster_id in clustered_data.keys():
        if len(clustered_data[cluster_id]) > max_cluster_dim:
            print("Program: " + bootqa_program)
            print("Test cases of cluster " + str(cluster_id) + ": " + str(len(clustered_data[cluster_id])))

    # Plotting the clusters in 3D space
    fig = plt.figure(figsize=(10, 7))
    if bootqa_program != "elevator2":
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection='3d')

    # Extracting data for plotting
    exec_costs = np.array(test_cases_costs)
    if bootqa_program != "elevator2":
        effectiveness = np.array(test_cases_effectiveness)
    else:
        pcounts = np.array(test_cases_pcount)
        dists = np.array(test_cases_dist)

    # Plot each refined cluster with a different color
    colors = plt.cm.get_cmap("tab10", len(clustered_data))  # A colormap with as many colors as clusters
    for cluster_id, members in clustered_data.items():
        if bootqa_program != "elevator2":
            ax.scatter(
                exec_costs[members],
                effectiveness[members],
                color=colors(cluster_id % 10),
                label=f"Cluster {cluster_id + 1}"
            )
        else:
            ax.scatter(
                exec_costs[members],
                pcounts[members],
                dists[members],
                color=colors(cluster_id % 10),
                label=f"Cluster {cluster_id + 1}"
            )

    # Label the axes
    ax.set_xlabel("Execution Cost")
    if bootqa_program != "elevator2":
        ax.set_ylabel("Effectiveness")
    else:
        ax.set_ylabel("Passengers Count")
        ax.set_zlabel("Travel Distance")
    ax.legend()
    ax.set_title("Test Case Clustering Visualization for: " + bootqa_program)

    # Display the plot
    # plt.show()


def make_linear_terms_bootqa(cluster_test_cases, test_cases_costs, test_cases_rates, alpha):
    """Making the linear terms of the QUBO"""
    max_cost = max(test_cases_costs)

    estimated_costs = []

    # linear coefficients, that are the diagonal of the matrix encoding the QUBO
    for test_case in cluster_test_cases:
        estimated_costs.append(
            (alpha * ((test_cases_costs[test_case]) / max_cost)) - ((1 - alpha) * test_cases_rates[test_case]))

    return np.array(estimated_costs)


def make_linear_terms_bootqa2(cluster_test_cases, test_cases_costs, pcount, dist, alpha, beta, gamma):
    """Making the linear terms of the QUBO for the elevator2 problem"""
    max_cost = max(test_cases_costs)
    max_pcount = max(pcount)
    max_dist = max(dist)

    estimated_costs = []

    # linear coefficients, that are the diagonal of the matrix encoding the QUBO
    for test_case in cluster_test_cases:
        estimated_costs.append(
            ((alpha) * ((test_cases_costs[test_case]) / max_cost)) - ((beta) * ((pcount[test_case]) / max_pcount)) - (
                        (gamma) * ((dist[test_case]) / max_dist)))

    return np.array(estimated_costs)


def create_linear_qubo(linear_terms):
    """This function is the one that has to encode the QUBO problem that QAOA will have to solve. The QUBO problem specifies the optimization to solve and a quadratic binary unconstrained problem"""
    qubo = QuadraticProgram()

    for i in range(0, len(linear_terms)):
        qubo.binary_var('x%s' % (i))

    qubo.minimize(linear=linear_terms)

    return qubo


def bootstrap_confidence_interval(data, num_samples, confidence_alpha=0.95):
    """This function determines the statistical range within we would expect the mean value of execution times to fall; it relies on the bootstrapping strategy, which allows the calculation of the confidence interval by repeated sampling (with replacement) from the existing data to obtain an estimate of the confidence interval."""
    sample_means = []
    for _ in range(num_samples):
        bootstrap_sample = [random.choice(data) for _ in range(len(data))]
        sample_mean = np.mean(bootstrap_sample)
        sample_means.append(sample_mean)

    lower_percentile = (1 - confidence_alpha) / 2 * 100
    upper_percentile = (confidence_alpha + (1 - confidence_alpha) / 2) * 100
    lower_bound = np.percentile(sample_means, lower_percentile)
    upper_bound = np.percentile(sample_means, upper_percentile)

    return lower_bound, upper_bound

def run_bootqa_with_alpha(bootqa_program, alpha, bootqa_clusters, experiments=3):
    """
    Run BootQA for a single program with given alpha(s) and return a scalar score.
    Lower score = better (Optuna minimizes by default).
    """
    ideal_sampler = AerSampler()
    ideal_sampler.options.shots = None
    rep = bootqa_programs_rep_values[bootqa_program]
    qaoa = QAOA(sampler=ideal_sampler, optimizer=COBYLA(500), reps=rep)

    data = get_data(bootqa_program)

    if bootqa_program in ("elevator", "elevator2"):
        test_cases_costs = data["cost"].tolist()
    else:
        test_cases_costs = data["time"].tolist()

    test_cases_effectiveness = None
    test_cases_pcount = None
    test_cases_dist = None

    if bootqa_program != "elevator2":
        test_cases_effectiveness = (
            data["input_div"].tolist() if bootqa_program == "elevator" else data["rate"].tolist()
        )
    else:
        test_cases_pcount = data["pcount"].tolist()
        test_cases_dist = data["dist"].tolist()

    all_costs = []
    all_effectiveness = []
    all_pcounts = []
    all_dists = []

    for _ in range(experiments):
        final_selected_cases = []

        for cluster_id in bootqa_clusters[bootqa_program]:
            cluster_members = bootqa_clusters[bootqa_program][cluster_id]

            if bootqa_program != "elevator2":
                linear_terms = make_linear_terms_bootqa(
                    cluster_members, test_cases_costs, test_cases_effectiveness, alpha
                )
            else:
                linear_terms = make_linear_terms_bootqa2(
                    cluster_members, test_cases_costs,
                    test_cases_pcount, test_cases_dist,
                    alpha[0], alpha[1], alpha[2]
                )

            linear_qubo = create_linear_qubo(linear_terms)
            operator, _ = linear_qubo.to_ising()
            qaoa_result = qaoa.compute_minimum_eigenvalue(operator)

            eigenstate = qaoa_result.eigenstate
            most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]

            n = linear_qubo.get_num_binary_vars()
            if isinstance(most_likely, int):
                bitstring = [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
            elif isinstance(most_likely, str):
                bitstring = [int(b) for b in most_likely[::-1]]
            else:
                raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")

            for idx, val in enumerate(bitstring):
                if val == 1:
                    tc = cluster_members[idx]
                    if tc not in final_selected_cases:
                        final_selected_cases.append(tc)

        # Aggregate metrics for this experiment
        total_cost = sum(test_cases_costs[tc] for tc in final_selected_cases)
        all_costs.append(total_cost)

        if bootqa_program != "elevator2":
            total_eff = sum(test_cases_effectiveness[tc] for tc in final_selected_cases)
            all_effectiveness.append(total_eff)
        else:
            all_pcounts.append(sum(test_cases_pcount[tc] for tc in final_selected_cases))
            all_dists.append(sum(test_cases_dist[tc] for tc in final_selected_cases))

    # --- Compute a scalar score to minimize ---
    # Normalize cost by max possible cost (full suite)
    max_possible_cost = sum(test_cases_costs)

    mean_cost = statistics.mean(all_costs)
    norm_cost = mean_cost / max_possible_cost  # in [0,1], lower is better

    if bootqa_program != "elevator2":
        max_possible_eff = sum(test_cases_effectiveness)
        mean_eff = statistics.mean(all_effectiveness)
        norm_eff = mean_eff / max_possible_eff  # in [0,1], higher is better

        # Score: minimize cost, maximize effectiveness
        # alpha already controls the trade-off, so we use equal weight here
        score = norm_cost - norm_eff

    else:
        max_possible_pcount = sum(test_cases_pcount)
        max_possible_dist = sum(test_cases_dist)
        mean_pcount = statistics.mean(all_pcounts)
        mean_dist = statistics.mean(all_dists)
        norm_pcount = mean_pcount / max_possible_pcount
        norm_dist = mean_dist / max_possible_dist

        # Score: minimize cost, maximize pcount and dist
        score = norm_cost - norm_pcount - norm_dist

    return score


def make_objective(bootqa_program, bootqa_clusters):
    """
    Returns an Optuna objective function for the given program.
    Closure captures program name and cluster structure.
    """
    def objective(trial):
        if bootqa_program != "elevator2":
            # Single alpha in (0, 1)
            alpha = trial.suggest_float("alpha", 0.0, 1.0)
        else:
            # Three alphas; constrained to sum to 1 for interpretability
            # (you can relax this if you prefer independent search)
            alpha_raw = [
                trial.suggest_float("alpha", 0.0, 1.0),
                trial.suggest_float("beta",  0.0, 1.0),
                trial.suggest_float("gamma", 0.0, 1.0),
            ]
            total = sum(alpha_raw) or 1.0          # avoid div-by-zero
            alpha = tuple(v / total for v in alpha_raw)  # normalize to sum=1

        score = run_bootqa_with_alpha(bootqa_program, alpha, bootqa_clusters, experiments=3)
        return score

    return objective


# ── Run Optuna for every program ──────────────────────────────────────────────
optuna_results = {}

for bootqa_program in bootqa_programs:
    print(f"\n{'='*60}")
    print(f"Optimizing alpha for: {bootqa_program}")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),   # Bayesian-style search
        pruner=optuna.pruners.MedianPruner(),
        study_name=f"bootqa_{bootqa_program}"
    )

    study.optimize(
        make_objective(bootqa_program, bootqa_clusters),
        n_trials=10,           # increase for a more thorough search
        show_progress_bar=True
    )

    best = study.best_trial
    optuna_results[bootqa_program] = {
        "best_params": best.params,
        "best_score":  best.value
    }

    print(f"\nBest params : {best.params}")
    print(f"Best score  : {best.value:.4f}")


# ── Summary ───────────────────────────────────────────────────────────────────
output_dir = "results/selectqaoa/statevector_sim"
os.makedirs(output_dir, exist_ok=True)

opt_params = {
    prog: result["best_params"]
    for prog, result in optuna_results.items()
}

file_path = os.path.join(output_dir, "so_opt_params.json")
with open(file_path, "w") as f:
    json.dump(opt_params, f, indent=2)

print(f"Optimal parameters saved to {file_path}")

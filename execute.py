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

from scipy.stats import mannwhitneyu, shapiro

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
from collections import defaultdict

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

if not callable(list):
    del list

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
        print(f"Tot suite cost: {sum(test_cases_costs)}")
        print(f"Tot suite input div: {sum(test_cases_effectiveness)}")
    elif bootqa_program == "elevator2":
        test_cases_pcount = data["pcount"].tolist()
        test_cases_dist = data["dist"].tolist()
        print(f"Tot suite cost: {sum(test_cases_costs)}")
        print(f"Tot suite pcount: {sum(test_cases_pcount)}")
        print(f"Tot suite dist: {sum(test_cases_dist)}")
    else:
        test_cases_effectiveness = data["rate"].tolist()
        print(f"Tot suite cost: {sum(test_cases_costs)}")
        print(f"Tot suite rate: {sum(test_cases_effectiveness)}")

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
    print("SelectQAOA Decomposition Time(ms): " + str((end - start) * 1000))

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
        print(f"Cluster {cluster_id + 1} metrics:")
        print(f"Test Cases: {members}")
        print(f" - Num. Test Cases: {len(members):.2f}")
        print(f" - Execution Cost: {tot_cluster_costs:.2f}")
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
    plt.show()


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


bootqa_alphas = {"gsdtsr": 0.2, "paintcontrol": 0.80, "iofrol": 0.82, "elevator": 0.50, "elevator2": (0.96, 0.96, 0.96)}
run_times_dictionary = {"gsdtsr": [], "paintcontrol": [], "iofrol": [], "elevator": [], "elevator2": []}
subsuites_results = {}
ideal_sampler = AerSampler()
ideal_sampler.options.shots = None

for bootqa_program in bootqa_programs:
    for rep in [1]:
        qaoa = QAOA(sampler=ideal_sampler, optimizer=COBYLA(500), reps=rep)
        data = get_data(bootqa_program)
        # Total suite metrics
        if bootqa_program != "elevator2" and bootqa_program != "elevator":
            test_cases_costs = data["time"].tolist()
        else:
            test_cases_costs = data["cost"].tolist()
        test_cases_effectiveness = None
        test_cases_pcount = None
        test_cases_dist = None
        if bootqa_program != "elevator2":
            test_cases_effectiveness = data["input_div"].tolist() if bootqa_program == "elevator" else data[
                "rate"].tolist()
        else:
            test_cases_pcount = data["pcount"].tolist()
            test_cases_dist = data["dist"].tolist()

        final_test_suite_costs = []
        final_effectivenesses = []
        final_pcounts = []
        final_dists = []
        for i in range(experiments):
            final_selected_cases = []
            cluster_number = 0
            for cluster_id in bootqa_clusters[bootqa_program]:
                print("Cluster: " + str(bootqa_clusters[bootqa_program][cluster_id]))
                linear_terms = None
                if bootqa_program != "elevator2":
                    linear_terms = make_linear_terms_bootqa(bootqa_clusters[bootqa_program][cluster_id],
                                                            test_cases_costs, test_cases_effectiveness,
                                                            bootqa_alphas[bootqa_program])
                else:
                    linear_terms = make_linear_terms_bootqa2(bootqa_clusters[bootqa_program][cluster_id],
                                                             test_cases_costs, test_cases_pcount, test_cases_dist,
                                                             bootqa_alphas[bootqa_program][0],
                                                             bootqa_alphas[bootqa_program][1],
                                                             bootqa_alphas[bootqa_program][2])
                linear_qubo = create_linear_qubo(linear_terms)
                operator, offset = linear_qubo.to_ising()
                print("Linear QUBO: " + str(linear_qubo))
                # for each iteration get the result
                s = time.time()
                qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
                e = time.time()
                # print("QAOA Result: " + str(qaoa_result))
                run_times_dictionary[bootqa_program].append((e - s) * 1000)

                eigenstate = qaoa_result.eigenstate
                most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]

                # Convert to bitstring format
                if isinstance(most_likely, int):
                    n = linear_qubo.get_num_binary_vars()
                    bitstring = [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
                elif isinstance(most_likely, str):
                    bitstring = [int(b) for b in most_likely[::-1]]
                else:
                    raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")

                indexes_selected_cases = [index for index, value in enumerate(bitstring) if value == 1]
                print("Indexes of selected tests to convert. " + str(indexes_selected_cases))
                selected_tests = []
                for index in indexes_selected_cases:
                    selected_tests.append(bootqa_clusters[bootqa_program][cluster_id][index])
                print("Selected tests: " + str(selected_tests))
                print(i)
                subsuites_results.setdefault(bootqa_program, {}).setdefault(i, {})[
                    str(cluster_id)] = linear_qubo.objective.evaluate(bitstring)
                print(subsuites_results)
                for selected_test in selected_tests:
                    if selected_test not in final_test_suite_costs:
                        final_selected_cases.append(selected_test)

            # compute the final test suite cost
            final_test_suite_cost = 0
            for selected_test_case in final_selected_cases:
                final_test_suite_cost += test_cases_costs[selected_test_case]
            final_test_suite_costs.append(final_test_suite_cost)

            # compute the total effectiveness
            if bootqa_program != "elevator2":
                final_effectiveness = 0
                for selected_test_case in final_selected_cases:
                    final_effectiveness += test_cases_effectiveness[selected_test_case]
                final_effectivenesses.append(final_effectiveness)
            else:
                final_pcount = 0
                for selected_test_case in final_selected_cases:
                    final_pcount += test_cases_pcount[selected_test_case]
                final_pcounts.append(final_pcount)

                final_dist = 0
                for selected_test_case in final_selected_cases:
                    final_dist += test_cases_dist[selected_test_case]
                final_dists.append(final_dist)

        print("Final Test Suite: " + str(final_selected_cases))
        # compute the qpu access times
        qpu_run_times_without_zeros = []
        for access_time in run_times_dictionary[bootqa_program]:
            if access_time != 0:
                qpu_run_times_without_zeros.append(access_time)
        lower_bound, upper_bound = bootstrap_confidence_interval(qpu_run_times_without_zeros, 1000, 0.95)
        for i in range(len(run_times_dictionary[bootqa_program])):
            if run_times_dictionary[bootqa_program][i] == 0:
                run_times_dictionary[bootqa_program][i] = upper_bound
        average_qpu_access_time = statistics.mean(run_times_dictionary[bootqa_program])

        if bootqa_program == "elevator2":
            var_names = ["final_test_suite_costs", "final_pcounts", "final_dists",
                         "average_qpu_access_time(ms)", "stdev_qpu_access_time(ms)", "all_qpu_access_times(ms)",
                         "qpu_lower_bound(ms)", "qpu_upper_bound(ms)", "qpu_run_times(ms)"]
            values = [final_test_suite_costs, final_pcounts, final_dists, average_qpu_access_time,
                      statistics.stdev(run_times_dictionary[bootqa_program]), run_times_dictionary[bootqa_program],
                      lower_bound, upper_bound, run_times_dictionary[bootqa_program]]
        else:
            var_names = ["final_test_suite_costs", "final_effectivenesses",
                         "average_qpu_access_time(ms)", "stdev_qpu_access_time(ms)", "all_qpu_access_times(ms)",
                         "qpu_lower_bound(ms)", "qpu_upper_bound(ms)", "qpu_run_times(ms)"]
            values = [final_test_suite_costs, final_effectivenesses, average_qpu_access_time,
                      statistics.stdev(run_times_dictionary[bootqa_program]), run_times_dictionary[bootqa_program],
                      lower_bound, upper_bound, run_times_dictionary[bootqa_program]]

        # Ensure the directory exists
        output_dir = "results/selectqaoa/statevector_sim"
        os.makedirs(output_dir, exist_ok=True)

        # Path to save the file
        file_path = os.path.join(output_dir, f"{bootqa_program}.csv")
        file_path_subsuites = os.path.join(output_dir, f"{bootqa_program}-subsuites.json")

        # Writing results to the file
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(var_names)
            writer.writerow(values)
        print(f"Results saved to {file_path}")

        with open(file_path_subsuites, "w") as file2:
            json.dump(subsuites_results, file2)

bootqa_alphas = {"gsdtsr": 0.2, "paintcontrol": 0.80, "iofrol": 0.82, "elevator": 0.50, "elevator2": (0.96, 0.96, 0.96)}
run_times_dictionary = {"gsdtsr": [], "paintcontrol": [], "iofrol": [], "elevator": [], "elevator2": []}
subsuites_results = {}
sampling_noise_sampler = AerSampler(backend_options={}, run_options={"shots": 2048})

for bootqa_program in bootqa_programs:
    qaoa = QAOA(sampler=sampling_noise_sampler, optimizer=COBYLA(500), reps=bootqa_programs_rep_values[bootqa_program])
    data = get_data(bootqa_program)
    # Total suite metrics
    if bootqa_program != "elevator2" and bootqa_program != "elevator":
        test_cases_costs = data["time"].tolist()
    else:
        test_cases_costs = data["cost"].tolist()
    test_cases_rates = None
    test_cases_pcount = None
    test_cases_dist = None
    if bootqa_program != "elevator2":
        test_cases_rates = data["input_div"].tolist() if bootqa_program == "elevator" else data["rate"].tolist()
    else:
        test_cases_pcount = data["pcount"].tolist()
        test_cases_dist = data["dist"].tolist()

    final_test_suite_costs = []
    final_effectivenesses = []
    final_pcounts = []
    final_dists = []
    for i in range(experiments):
        final_selected_cases = []
        cluster_number = 0
        for cluster_id in bootqa_clusters[bootqa_program]:
            print("Cluster: " + str(bootqa_clusters[bootqa_program][cluster_id]))
            linear_terms = None
            if bootqa_program != "elevator2":
                linear_terms = make_linear_terms_bootqa(bootqa_clusters[bootqa_program][cluster_id], test_cases_costs,
                                                        test_cases_rates, bootqa_alphas[bootqa_program])
            else:
                linear_terms = make_linear_terms_bootqa2(bootqa_clusters[bootqa_program][cluster_id], test_cases_costs,
                                                         test_cases_pcount, test_cases_dist,
                                                         bootqa_alphas[bootqa_program][0],
                                                         bootqa_alphas[bootqa_program][1],
                                                         bootqa_alphas[bootqa_program][2])
            linear_qubo = create_linear_qubo(linear_terms)
            operator, offset = linear_qubo.to_ising()
            print("Linear QUBO: " + str(linear_qubo))
            # for each iteration get the result
            s = time.time()
            qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
            e = time.time()
            print("optimal:" + str(qaoa_result.optimal_point))
            print("Exp: " + str(i))
            run_times_dictionary[bootqa_program].append((e - s) * 1000)

            eigenstate = qaoa_result.eigenstate
            most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]

            # Convert to bitstring format
            if isinstance(most_likely, int):
                n = linear_qubo.get_num_binary_vars()
                bitstring = [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
            elif isinstance(most_likely, str):
                bitstring = [int(b) for b in most_likely[::-1]]
            else:
                raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")
            indexes_selected_cases = [index for index, value in enumerate(bitstring) if value == 1]
            print("Indexes of selected tests to convert. " + str(indexes_selected_cases))
            selected_tests = []
            for index in indexes_selected_cases:
                selected_tests.append(bootqa_clusters[bootqa_program][cluster_id][index])
            print("Selected tests: " + str(selected_tests))
            print(i)
            subsuites_results.setdefault(bootqa_program, {}).setdefault(i, {})[
                str(cluster_id)] = linear_qubo.objective.evaluate(bitstring)
            print(subsuites_results)
            for selected_test in selected_tests:
                if selected_test not in final_test_suite_costs:
                    final_selected_cases.append(selected_test)

        # compute the final test suite cost
        final_test_suite_cost = 0
        for selected_test_case in final_selected_cases:
            final_test_suite_cost += test_cases_costs[selected_test_case]
        final_test_suite_costs.append(final_test_suite_cost)

        # compute the total failure rate
        if bootqa_program != "elevator2":
            final_effectiveness = 0
            for selected_test_case in final_selected_cases:
                final_effectiveness += test_cases_rates[selected_test_case]
            final_effectivenesses.append(final_effectiveness)
        else:
            final_pcount = 0
            for selected_test_case in final_selected_cases:
                final_pcount += test_cases_pcount[selected_test_case]
            final_pcounts.append(final_pcount)

            final_dist = 0
            for selected_test_case in final_selected_cases:
                final_dist += test_cases_dist[selected_test_case]
            final_dists.append(final_dist)

    print("Final Test Suite: " + str(final_selected_cases))
    # compute the qpu access times
    qpu_run_times_without_zeros = []
    for access_time in run_times_dictionary[bootqa_program]:
        if access_time != 0:
            qpu_run_times_without_zeros.append(access_time)
    lower_bound, upper_bound = bootstrap_confidence_interval(qpu_run_times_without_zeros, 1000, 0.95)
    for i in range(len(run_times_dictionary[bootqa_program])):
        if run_times_dictionary[bootqa_program][i] == 0:
            run_times_dictionary[bootqa_program][i] = upper_bound
    average_qpu_access_time = statistics.mean(run_times_dictionary[bootqa_program])

    if bootqa_program == "elevator2":
        var_names = ["final_test_suite_costs", "final_pcounts", "final_dists",
                     "average_qpu_access_time(ms)", "stdev_qpu_access_time(ms)", "all_qpu_access_times(ms)",
                     "qpu_lower_bound(ms)", "qpu_upper_bound(ms)", "qpu_run_times(ms)"]
        values = [final_test_suite_costs, final_pcounts, final_dists, average_qpu_access_time,
                  statistics.stdev(run_times_dictionary[bootqa_program]), run_times_dictionary[bootqa_program],
                  lower_bound, upper_bound, run_times_dictionary[bootqa_program]]
    else:
        var_names = ["final_test_suite_costs", "final_effectivenesses",
                     "average_qpu_access_time(ms)", "stdev_qpu_access_time(ms)", "all_qpu_access_times(ms)",
                     "qpu_lower_bound(ms)", "qpu_upper_bound(ms)", "qpu_run_times(ms)"]
        values = [final_test_suite_costs, final_effectivenesses, average_qpu_access_time,
                  statistics.stdev(run_times_dictionary[bootqa_program]), run_times_dictionary[bootqa_program],
                  lower_bound, upper_bound, run_times_dictionary[bootqa_program]]

    # Ensure the directory exists
    output_dir = "results/selectqaoa/aer_sim"
    os.makedirs(output_dir, exist_ok=True)

    # Path to save the file
    file_path = os.path.join(output_dir, f"{bootqa_program}.csv")
    file_path_subsuites = os.path.join(output_dir, f"{bootqa_program}-subsuites.json")

    # Writing results to the file
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(var_names)
        writer.writerow(values)
    print(f"Results saved to {file_path}")

    with open(file_path_subsuites, "w") as file2:
        json.dump(subsuites_results, file2)

bootqa_alphas = {"gsdtsr": 0.2, "paintcontrol": 0.80, "iofrol": 0.82, "elevator": 0.50, "elevator2": (0.96, 0.96, 0.96)}
run_times_dictionary = {"gsdtsr": [], "paintcontrol": [], "iofrol": [], "elevator": [], "elevator2": []}
subsuites_results = {}
noise_model = NoiseModel.from_backend(FakeBrisbane())
fake_sampler = AerSampler(backend_options={'noise_model': noise_model})
fake_sampler.options.shots = 2048

for bootqa_program in bootqa_programs:
    qaoa = QAOA(sampler=fake_sampler, optimizer=COBYLA(500), reps=bootqa_programs_rep_values[bootqa_program])
    data = get_data(bootqa_program)
    # Total suite metrics
    if bootqa_program != "elevator2" and bootqa_program != "elevator":
        test_cases_costs = data["time"].tolist()
    else:
        test_cases_costs = data["cost"].tolist()
    test_cases_rates = None
    test_cases_pcount = None
    test_cases_dist = None
    if bootqa_program != "elevator2":
        test_cases_rates = data["input_div"].tolist() if bootqa_program == "elevator" else data["rate"].tolist()
    else:
        test_cases_pcount = data["pcount"].tolist()
        test_cases_dist = data["dist"].tolist()

    final_test_suite_costs = []
    final_effectivenesses = []
    final_pcounts = []
    final_dists = []
    for i in range(experiments):
        final_selected_cases = []
        cluster_number = 0
        for cluster_id in bootqa_clusters[bootqa_program]:
            print("Cluster: " + str(bootqa_clusters[bootqa_program][cluster_id]))
            linear_terms = None
            if bootqa_program != "elevator2":
                linear_terms = make_linear_terms_bootqa(bootqa_clusters[bootqa_program][cluster_id], test_cases_costs,
                                                        test_cases_rates, bootqa_alphas[bootqa_program])
            else:
                linear_terms = make_linear_terms_bootqa2(bootqa_clusters[bootqa_program][cluster_id], test_cases_costs,
                                                         test_cases_pcount, test_cases_dist,
                                                         bootqa_alphas[bootqa_program][0],
                                                         bootqa_alphas[bootqa_program][1],
                                                         bootqa_alphas[bootqa_program][2])
            linear_qubo = create_linear_qubo(linear_terms)
            operator, offset = linear_qubo.to_ising()
            print("Linear QUBO: " + str(linear_qubo))
            # for each iteration get the result
            s = time.time()
            qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
            e = time.time()
            # print("QAOA Result: " + str(qaoa_result))
            run_times_dictionary[bootqa_program].append((e - s) * 1000)

            eigenstate = qaoa_result.eigenstate
            most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]

            # Convert to bitstring format
            if isinstance(most_likely, int):
                n = linear_qubo.get_num_binary_vars()
                bitstring = [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
            elif isinstance(most_likely, str):
                bitstring = [int(b) for b in most_likely[::-1]]
            else:
                raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")
            indexes_selected_cases = [index for index, value in enumerate(bitstring) if value == 1]
            print("Indexes of selected tests to convert. " + str(indexes_selected_cases))
            selected_tests = []
            for index in indexes_selected_cases:
                selected_tests.append(bootqa_clusters[bootqa_program][cluster_id][index])
            print("Selected tests: " + str(selected_tests))
            subsuites_results.setdefault(bootqa_program, {}).setdefault(i, {})[
                str(cluster_id)] = linear_qubo.objective.evaluate(bitstring)
            for selected_test in selected_tests:
                if selected_test not in final_test_suite_costs:
                    final_selected_cases.append(selected_test)

        # compute the final test suite cost
        final_test_suite_cost = 0
        for selected_test_case in final_selected_cases:
            final_test_suite_cost += test_cases_costs[selected_test_case]
        final_test_suite_costs.append(final_test_suite_cost)

        # compute the total failure rate
        if bootqa_program != "elevator2":
            final_effectiveness = 0
            for selected_test_case in final_selected_cases:
                final_effectiveness += test_cases_rates[selected_test_case]
            final_effectivenesses.append(final_effectiveness)
        else:
            final_pcount = 0
            for selected_test_case in final_selected_cases:
                final_pcount += test_cases_pcount[selected_test_case]
            final_pcounts.append(final_pcount)

            final_dist = 0
            for selected_test_case in final_selected_cases:
                final_dist += test_cases_dist[selected_test_case]
            final_dists.append(final_dist)

    print("Final Test Suite: " + str(final_selected_cases))
    # compute the qpu access times
    qpu_run_times_without_zeros = []
    for access_time in run_times_dictionary[bootqa_program]:
        if access_time != 0:
            qpu_run_times_without_zeros.append(access_time)
    lower_bound, upper_bound = bootstrap_confidence_interval(qpu_run_times_without_zeros, 1000, 0.95)
    for i in range(len(run_times_dictionary[bootqa_program])):
        if run_times_dictionary[bootqa_program][i] == 0:
            run_times_dictionary[bootqa_program][i] = upper_bound
    average_qpu_access_time = statistics.mean(run_times_dictionary[bootqa_program])

    if bootqa_program == "elevator2":
        var_names = ["final_test_suite_costs", "final_pcounts", "final_dists",
                     "average_qpu_access_time(ms)", "stdev_qpu_access_time(ms)", "all_qpu_access_times(ms)",
                     "qpu_lower_bound(ms)", "qpu_upper_bound(ms)", "qpu_run_times(ms)"]
        values = [final_test_suite_costs, final_pcounts, final_dists, average_qpu_access_time,
                  statistics.stdev(run_times_dictionary[bootqa_program]), run_times_dictionary[bootqa_program],
                  lower_bound, upper_bound, run_times_dictionary[bootqa_program]]
    else:
        var_names = ["final_test_suite_costs", "final_effectivenesses",
                     "average_qpu_access_time(ms)", "stdev_qpu_access_time(ms)", "all_qpu_access_times(ms)",
                     "qpu_lower_bound(ms)", "qpu_upper_bound(ms)", "qpu_run_times(ms)"]
        values = [final_test_suite_costs, final_effectivenesses, average_qpu_access_time,
                  statistics.stdev(run_times_dictionary[bootqa_program]), run_times_dictionary[bootqa_program],
                  lower_bound, upper_bound, run_times_dictionary[bootqa_program]]

    # Ensure the directory exists
    output_dir = "results/selectqaoa/fake_brisbane"
    os.makedirs(output_dir, exist_ok=True)

    # Path to save the file
    file_path = os.path.join(output_dir, f"{bootqa_program}.csv")
    file_path_subsuites = os.path.join(output_dir, f"{bootqa_program}-subsuites.json")

    # Writing results to the file
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(var_names)
        writer.writerow(values)
    print(f"Results saved to {file_path}")

    with open(file_path_subsuites, "w") as file2:
        json.dump(subsuites_results, file2)

bootqa_alphas = {"gsdtsr": 0.2, "paintcontrol": 0.80, "iofrol": 0.82, "elevator": 0.50, "elevator2": (0.96, 0.96, 0.96)}
run_times_dictionary = {"gsdtsr": [], "paintcontrol": [], "iofrol": [], "elevator": [], "elevator2": []}
params = {}
subsuites_results = {}
noise_model = NoiseModel()
error_1 = depolarizing_error(0.01, 1)
error_2 = depolarizing_error(0.01, 2)
noise_model.add_all_qubit_quantum_error(error_1, ['x', 'y', 'z', 'h', 'ry', 'rz', 'rx', 'sx', 'id'])
noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

noisy_sampler = AerSampler(backend_options={'noise_model': noise_model})
noisy_sampler.options.shots = 2048

for bootqa_program in bootqa_programs:
    qaoa = QAOA(sampler=noisy_sampler, optimizer=COBYLA(500), reps=bootqa_programs_rep_values[bootqa_program])
    data = get_data(bootqa_program)
    # Total suite metrics
    if bootqa_program != "elevator2" and bootqa_program != "elevator":
        test_cases_costs = data["time"].tolist()
    else:
        test_cases_costs = data["cost"].tolist()
    test_cases_rates = None
    test_cases_pcount = None
    test_cases_dist = None
    if bootqa_program != "elevator2":
        test_cases_rates = data["input_div"].tolist() if bootqa_program == "elevator" else data["rate"].tolist()
    else:
        test_cases_pcount = data["pcount"].tolist()
        test_cases_dist = data["dist"].tolist()

    final_test_suite_costs = []
    final_effectivenesses = []
    final_pcounts = []
    final_dists = []
    for i in range(experiments):
        final_selected_cases = []
        cluster_number = 0
        for cluster_id in bootqa_clusters[bootqa_program]:
            print("Cluster: " + str(bootqa_clusters[bootqa_program][cluster_id]))
            linear_terms = None
            if bootqa_program != "elevator2":
                linear_terms = make_linear_terms_bootqa(bootqa_clusters[bootqa_program][cluster_id], test_cases_costs,
                                                        test_cases_rates, bootqa_alphas[bootqa_program])
            else:
                linear_terms = make_linear_terms_bootqa2(bootqa_clusters[bootqa_program][cluster_id], test_cases_costs,
                                                         test_cases_pcount, test_cases_dist,
                                                         bootqa_alphas[bootqa_program][0],
                                                         bootqa_alphas[bootqa_program][1],
                                                         bootqa_alphas[bootqa_program][2])
            linear_qubo = create_linear_qubo(linear_terms)
            operator, offset = linear_qubo.to_ising()
            print("Linear QUBO: " + str(linear_qubo))
            # for each iteration get the result
            s = time.time()
            qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
            e = time.time()
            params.setdefault(bootqa_program, {}).setdefault(i, {})[cluster_id] = qaoa_result.optimal_point.tolist()
            print(params)
            print(i)
            # print("QAOA Result: " + str(qaoa_result))
            run_times_dictionary[bootqa_program].append((e - s) * 1000)

            eigenstate = qaoa_result.eigenstate
            most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]

            # Convert to bitstring format
            if isinstance(most_likely, int):
                n = linear_qubo.get_num_binary_vars()
                bitstring = [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
            elif isinstance(most_likely, str):
                bitstring = [int(b) for b in most_likely[::-1]]
            else:
                raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")
            indexes_selected_cases = [index for index, value in enumerate(bitstring) if value == 1]
            print("Indexes of selected tests to convert. " + str(indexes_selected_cases))
            selected_tests = []
            for index in indexes_selected_cases:
                selected_tests.append(bootqa_clusters[bootqa_program][cluster_id][index])
            print(i)
            print("Selected tests: " + str(selected_tests))
            subsuites_results.setdefault(bootqa_program, {}).setdefault(i, {})[
                str(cluster_id)] = linear_qubo.objective.evaluate(bitstring)
            print(subsuites_results)
            for selected_test in selected_tests:
                if selected_test not in final_test_suite_costs:
                    final_selected_cases.append(selected_test)

        # compute the final test suite cost
        final_test_suite_cost = 0
        for selected_test_case in final_selected_cases:
            final_test_suite_cost += test_cases_costs[selected_test_case]
        final_test_suite_costs.append(final_test_suite_cost)

        # compute the total failure rate
        if bootqa_program != "elevator2":
            final_effectiveness = 0
            for selected_test_case in final_selected_cases:
                final_effectiveness += test_cases_rates[selected_test_case]
            final_effectivenesses.append(final_effectiveness)
        else:
            final_pcount = 0
            for selected_test_case in final_selected_cases:
                final_pcount += test_cases_pcount[selected_test_case]
            final_pcounts.append(final_pcount)

            final_dist = 0
            for selected_test_case in final_selected_cases:
                final_dist += test_cases_dist[selected_test_case]
            final_dists.append(final_dist)

    print("Final Test Suite: " + str(final_selected_cases))
    # compute the qpu access times
    qpu_run_times_without_zeros = []
    for access_time in run_times_dictionary[bootqa_program]:
        if access_time != 0:
            qpu_run_times_without_zeros.append(access_time)
    lower_bound, upper_bound = bootstrap_confidence_interval(qpu_run_times_without_zeros, 1000, 0.95)
    for i in range(len(run_times_dictionary[bootqa_program])):
        if run_times_dictionary[bootqa_program][i] == 0:
            run_times_dictionary[bootqa_program][i] = upper_bound
    average_qpu_access_time = statistics.mean(run_times_dictionary[bootqa_program])

    if bootqa_program == "elevator2":
        var_names = ["final_test_suite_costs", "final_pcounts", "final_dists",
                     "average_qpu_access_time(ms)", "stdev_qpu_access_time(ms)", "all_qpu_access_times(ms)",
                     "qpu_lower_bound(ms)", "qpu_upper_bound(ms)", "qpu_run_times(ms)"]
        values = [final_test_suite_costs, final_pcounts, final_dists, average_qpu_access_time,
                  statistics.stdev(run_times_dictionary[bootqa_program]), run_times_dictionary[bootqa_program],
                  lower_bound, upper_bound, run_times_dictionary[bootqa_program]]
    else:
        var_names = ["final_test_suite_costs", "final_effectivenesses",
                     "average_qpu_access_time(ms)", "stdev_qpu_access_time(ms)", "all_qpu_access_times(ms)",
                     "qpu_lower_bound(ms)", "qpu_upper_bound(ms)", "qpu_run_times(ms)"]
        values = [final_test_suite_costs, final_effectivenesses, average_qpu_access_time,
                  statistics.stdev(run_times_dictionary[bootqa_program]), run_times_dictionary[bootqa_program],
                  lower_bound, upper_bound, run_times_dictionary[bootqa_program]]

    # Ensure the directory exists
    output_dir = "results/selectqaoa/depolarizing_sim/01"
    os.makedirs(output_dir, exist_ok=True)

    # Path to save the file
    file_path = os.path.join(output_dir, f"{bootqa_program}.csv")
    file_path_subsuites = os.path.join(output_dir, f"{bootqa_program}-subsuites.json")

    # Writing results to the file
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(var_names)
        writer.writerow(values)
    print(f"Results saved to {file_path}")

    with open(file_path_subsuites, "w") as file2:
        json.dump(subsuites_results, file2)

bootqa_alphas = {"gsdtsr": 0.2, "paintcontrol": 0.80, "iofrol": 0.82, "elevator": 0.50, "elevator2": (0.96, 0.96, 0.96)}
run_times_dictionary = {"gsdtsr": [], "paintcontrol": [], "iofrol": [], "elevator": [], "elevator2": []}
subsuites_results = {}
noise_model = NoiseModel()
error_1 = depolarizing_error(0.02, 1)
error_2 = depolarizing_error(0.02, 2)
noise_model.add_all_qubit_quantum_error(error_1, ['x', 'y', 'z', 'h', 'ry', 'rz', 'rx', 'sx', 'id'])
noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

noisy_sampler = AerSampler(backend_options={'noise_model': noise_model})
noisy_sampler.options.shots = 2048

for bootqa_program in bootqa_programs:
    qaoa = QAOA(sampler=noisy_sampler, optimizer=COBYLA(500),
                reps=bootqa_programs_rep_values[bootqa_program])
    data = get_data(bootqa_program)
    # Total suite metrics
    if bootqa_program != "elevator2" and bootqa_program != "elevator":
        test_cases_costs = data["time"].tolist()
    else:
        test_cases_costs = data["cost"].tolist()
    test_cases_rates = None
    test_cases_pcount = None
    test_cases_dist = None
    if bootqa_program != "elevator2":
        test_cases_rates = data["input_div"].tolist() if bootqa_program == "elevator" else data["rate"].tolist()
    else:
        test_cases_pcount = data["pcount"].tolist()
        test_cases_dist = data["dist"].tolist()

    final_test_suite_costs = []
    final_effectivenesses = []
    final_pcounts = []
    final_dists = []
    for i in range(experiments):
        final_selected_cases = []
        cluster_number = 0
        for cluster_id in bootqa_clusters[bootqa_program]:
            print("Cluster: " + str(bootqa_clusters[bootqa_program][cluster_id]))
            linear_terms = None
            if bootqa_program != "elevator2":
                linear_terms = make_linear_terms_bootqa(bootqa_clusters[bootqa_program][cluster_id], test_cases_costs,
                                                        test_cases_rates, bootqa_alphas[bootqa_program])
            else:
                linear_terms = make_linear_terms_bootqa2(bootqa_clusters[bootqa_program][cluster_id], test_cases_costs,
                                                         test_cases_pcount, test_cases_dist,
                                                         bootqa_alphas[bootqa_program][0],
                                                         bootqa_alphas[bootqa_program][1],
                                                         bootqa_alphas[bootqa_program][2])
            linear_qubo = create_linear_qubo(linear_terms)
            operator, offset = linear_qubo.to_ising()
            print("Linear QUBO: " + str(linear_qubo))
            # for each iteration get the result
            s = time.time()
            qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
            e = time.time()
            # print("QAOA Result: " + str(qaoa_result))
            run_times_dictionary[bootqa_program].append((e - s) * 1000)

            eigenstate = qaoa_result.eigenstate
            most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]

            # Convert to bitstring format
            if isinstance(most_likely, int):
                n = linear_qubo.get_num_binary_vars()
                bitstring = [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
            elif isinstance(most_likely, str):
                bitstring = [int(b) for b in most_likely[::-1]]
            else:
                raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")
            indexes_selected_cases = [index for index, value in enumerate(bitstring) if value == 1]
            print("Indexes of selected tests to convert. " + str(indexes_selected_cases))
            selected_tests = []
            for index in indexes_selected_cases:
                selected_tests.append(bootqa_clusters[bootqa_program][cluster_id][index])
            print("Selected tests: " + str(selected_tests))
            subsuites_results.setdefault(bootqa_program, {}).setdefault(i, {})[
                str(cluster_id)] = linear_qubo.objective.evaluate(bitstring)
            for selected_test in selected_tests:
                if selected_test not in final_test_suite_costs:
                    final_selected_cases.append(selected_test)

        # compute the final test suite cost
        final_test_suite_cost = 0
        for selected_test_case in final_selected_cases:
            final_test_suite_cost += test_cases_costs[selected_test_case]
        final_test_suite_costs.append(final_test_suite_cost)

        # compute the total failure rate
        if bootqa_program != "elevator2":
            final_effectiveness = 0
            for selected_test_case in final_selected_cases:
                final_effectiveness += test_cases_rates[selected_test_case]
            final_effectivenesses.append(final_effectiveness)
        else:
            final_pcount = 0
            for selected_test_case in final_selected_cases:
                final_pcount += test_cases_pcount[selected_test_case]
            final_pcounts.append(final_pcount)

            final_dist = 0
            for selected_test_case in final_selected_cases:
                final_dist += test_cases_dist[selected_test_case]
            final_dists.append(final_dist)

    print("Final Test Suite: " + str(final_selected_cases))
    # compute the qpu access times
    qpu_run_times_without_zeros = []
    for access_time in run_times_dictionary[bootqa_program]:
        if access_time != 0:
            qpu_run_times_without_zeros.append(access_time)
    lower_bound, upper_bound = bootstrap_confidence_interval(qpu_run_times_without_zeros, 1000, 0.95)
    for i in range(len(run_times_dictionary[bootqa_program])):
        if run_times_dictionary[bootqa_program][i] == 0:
            run_times_dictionary[bootqa_program][i] = upper_bound
    average_qpu_access_time = statistics.mean(run_times_dictionary[bootqa_program])

    if bootqa_program == "elevator2":
        var_names = ["final_test_suite_costs", "final_pcounts", "final_dists",
                     "average_qpu_access_time(ms)", "stdev_qpu_access_time(ms)", "all_qpu_access_times(ms)",
                     "qpu_lower_bound(ms)", "qpu_upper_bound(ms)", "qpu_run_times(ms)"]
        values = [final_test_suite_costs, final_pcounts, final_dists, average_qpu_access_time,
                  statistics.stdev(run_times_dictionary[bootqa_program]), run_times_dictionary[bootqa_program],
                  lower_bound, upper_bound, run_times_dictionary[bootqa_program]]
    else:
        var_names = ["final_test_suite_costs", "final_effectivenesses",
                     "average_qpu_access_time(ms)", "stdev_qpu_access_time(ms)", "all_qpu_access_times(ms)",
                     "qpu_lower_bound(ms)", "qpu_upper_bound(ms)", "qpu_run_times(ms)"]
        values = [final_test_suite_costs, final_effectivenesses, average_qpu_access_time,
                  statistics.stdev(run_times_dictionary[bootqa_program]), run_times_dictionary[bootqa_program],
                  lower_bound, upper_bound, run_times_dictionary[bootqa_program]]

    # Ensure the directory exists
    output_dir = "results/selectqaoa/depolarizing_sim/02"
    os.makedirs(output_dir, exist_ok=True)

    # Path to save the file
    file_path = os.path.join(output_dir, f"{bootqa_program}.csv")
    file_path_subsuites = os.path.join(output_dir, f"{bootqa_program}-subsuites.json")

    # Writing results to the file
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(var_names)
        writer.writerow(values)
    print(f"Results saved to {file_path}")

    with open(file_path_subsuites, "w") as file2:
        json.dump(subsuites_results, file2)

bootqa_alphas = {"gsdtsr": 0.2, "paintcontrol": 0.80, "iofrol": 0.82, "elevator": 0.50, "elevator2": (0.96, 0.96, 0.96)}
run_times_dictionary = {"gsdtsr": [], "paintcontrol": [], "iofrol": [], "elevator": [], "elevator2": []}
subsuites_results = {}
noise_model = NoiseModel()
error_1 = depolarizing_error(0.05, 1)
error_2 = depolarizing_error(0.05, 2)
noise_model.add_all_qubit_quantum_error(error_1, ['x', 'y', 'z', 'h', 'ry', 'rz', 'rx', 'sx', 'id'])
noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

noisy_sampler = AerSampler(backend_options={'noise_model': noise_model})
noisy_sampler.options.shots = 2048

for bootqa_program in bootqa_programs:
    qaoa = QAOA(sampler=noisy_sampler, optimizer=COBYLA(500),
                reps=bootqa_programs_rep_values[bootqa_program])
    data = get_data(bootqa_program)
    # Total suite metrics
    if bootqa_program != "elevator2" and bootqa_program != "elevator":
        test_cases_costs = data["time"].tolist()
    else:
        test_cases_costs = data["cost"].tolist()
    test_cases_rates = None
    test_cases_pcount = None
    test_cases_dist = None
    if bootqa_program != "elevator2":
        test_cases_rates = data["input_div"].tolist() if bootqa_program == "elevator" else data["rate"].tolist()
    else:
        test_cases_pcount = data["pcount"].tolist()
        test_cases_dist = data["dist"].tolist()

    final_test_suite_costs = []
    final_effectivenesses = []
    final_pcounts = []
    final_dists = []
    for i in range(experiments):
        final_selected_cases = []
        cluster_number = 0
        for cluster_id in bootqa_clusters[bootqa_program]:
            print("Cluster: " + str(bootqa_clusters[bootqa_program][cluster_id]))
            linear_terms = None
            if bootqa_program != "elevator2":
                linear_terms = make_linear_terms_bootqa(bootqa_clusters[bootqa_program][cluster_id], test_cases_costs,
                                                        test_cases_rates, bootqa_alphas[bootqa_program])
            else:
                linear_terms = make_linear_terms_bootqa2(bootqa_clusters[bootqa_program][cluster_id], test_cases_costs,
                                                         test_cases_pcount, test_cases_dist,
                                                         bootqa_alphas[bootqa_program][0],
                                                         bootqa_alphas[bootqa_program][1],
                                                         bootqa_alphas[bootqa_program][2])
            linear_qubo = create_linear_qubo(linear_terms)
            operator, offset = linear_qubo.to_ising()
            print("Linear QUBO: " + str(linear_qubo))
            # for each iteration get the result
            s = time.time()
            qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
            e = time.time()
            # print("QAOA Result: " + str(qaoa_result))
            run_times_dictionary[bootqa_program].append((e - s) * 1000)

            eigenstate = qaoa_result.eigenstate
            most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]

            # Convert to bitstring format
            if isinstance(most_likely, int):
                n = linear_qubo.get_num_binary_vars()
                bitstring = [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
            elif isinstance(most_likely, str):
                bitstring = [int(b) for b in most_likely[::-1]]
            else:
                raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")
            indexes_selected_cases = [index for index, value in enumerate(bitstring) if value == 1]
            print("Indexes of selected tests to convert. " + str(indexes_selected_cases))
            selected_tests = []
            for index in indexes_selected_cases:
                selected_tests.append(bootqa_clusters[bootqa_program][cluster_id][index])
            print("Selected tests: " + str(selected_tests))
            subsuites_results.setdefault(bootqa_program, {}).setdefault(i, {})[
                str(cluster_id)] = linear_qubo.objective.evaluate(bitstring)
            for selected_test in selected_tests:
                if selected_test not in final_test_suite_costs:
                    final_selected_cases.append(selected_test)

        # compute the final test suite cost
        final_test_suite_cost = 0
        for selected_test_case in final_selected_cases:
            final_test_suite_cost += test_cases_costs[selected_test_case]
        final_test_suite_costs.append(final_test_suite_cost)

        # compute the total failure rate
        if bootqa_program != "elevator2":
            final_effectiveness = 0
            for selected_test_case in final_selected_cases:
                final_effectiveness += test_cases_rates[selected_test_case]
            final_effectivenesses.append(final_effectiveness)
        else:
            final_pcount = 0
            for selected_test_case in final_selected_cases:
                final_pcount += test_cases_pcount[selected_test_case]
            final_pcounts.append(final_pcount)

            final_dist = 0
            for selected_test_case in final_selected_cases:
                final_dist += test_cases_dist[selected_test_case]
            final_dists.append(final_dist)

    print("Final Test Suite: " + str(final_selected_cases))
    # compute the qpu access times
    qpu_run_times_without_zeros = []
    for access_time in run_times_dictionary[bootqa_program]:
        if access_time != 0:
            qpu_run_times_without_zeros.append(access_time)
    lower_bound, upper_bound = bootstrap_confidence_interval(qpu_run_times_without_zeros, 1000, 0.95)
    for i in range(len(run_times_dictionary[bootqa_program])):
        if run_times_dictionary[bootqa_program][i] == 0:
            run_times_dictionary[bootqa_program][i] = upper_bound
    average_qpu_access_time = statistics.mean(run_times_dictionary[bootqa_program])

    if bootqa_program == "elevator2":
        var_names = ["final_test_suite_costs", "final_pcounts", "final_dists",
                     "average_qpu_access_time(ms)", "stdev_qpu_access_time(ms)", "all_qpu_access_times(ms)",
                     "qpu_lower_bound(ms)", "qpu_upper_bound(ms)", "qpu_run_times(ms)"]
        values = [final_test_suite_costs, final_pcounts, final_dists, average_qpu_access_time,
                  statistics.stdev(run_times_dictionary[bootqa_program]), run_times_dictionary[bootqa_program],
                  lower_bound, upper_bound, run_times_dictionary[bootqa_program]]
    else:
        var_names = ["final_test_suite_costs", "final_effectivenesses",
                     "average_qpu_access_time(ms)", "stdev_qpu_access_time(ms)", "all_qpu_access_times(ms)",
                     "qpu_lower_bound(ms)", "qpu_upper_bound(ms)", "qpu_run_times(ms)"]
        values = [final_test_suite_costs, final_effectivenesses, average_qpu_access_time,
                  statistics.stdev(run_times_dictionary[bootqa_program]), run_times_dictionary[bootqa_program],
                  lower_bound, upper_bound, run_times_dictionary[bootqa_program]]

    # Ensure the directory exists
    output_dir = "results/selectqaoa/depolarizing_sim/05"
    os.makedirs(output_dir, exist_ok=True)

    # Path to save the file
    file_path = os.path.join(output_dir, f"{bootqa_program}.csv")
    file_path_subsuites = os.path.join(output_dir, f"{bootqa_program}-subsuites.json")

    # Writing results to the file
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(var_names)
        writer.writerow(values)
    print(f"Results saved to {file_path}")

    with open(file_path_subsuites, "w") as file2:
        json.dump(subsuites_results, file2)


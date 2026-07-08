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

from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

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

#this cell contains all variable definitions that will be useful throughout the entire project
sir_programs = ["flex","grep","gzip","sed"]
sir_programs_tests_number = {"flex":567,"grep":806,"gzip":214,"sed":360}
sir_programs_end_lines = {"flex":14192,"grep":13281,"gzip":6701,"sed":7118}
sir_programs_rep_values = {"flex":1,"grep":1,"gzip":1,"sed":1}
alpha = 0.5
experiments = 10

def json_keys_to_int(d):
    """This method correctly converts the data"""
    if isinstance(d, dict):
        return {int(k) if k.isdigit() else k: json_keys_to_int(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [json_keys_to_int(i) for i in d]
    else:
        return d


with open("datasets/sir_programs/executed_lines_test_by_test.json", "r") as file:
    #dictionary that, for each sir program, associates at each LINE of that program the LIST of TESTS COVERING it
    executed_lines_test_by_test = json_keys_to_int(json.load(file)) #{program1:{line:[tci,tcj,...,tck],line2:...}
with open("datasets/sir_programs/faults_dictionary.json", "r") as file:
    #dictionary that associates at each SIR PROGRAM the LIST of PAST FAULT COVERAGE VALUES ORDERED BY TEST
    faults_dictionary = json.load(file) #{program1:[fault_cov_tc1,fault_cov_tc2,...,fault_cov_tcn],program2:...}
with open("datasets/sir_programs/test_coverage_line_by_line.json", "r") as file:
    #dictionary that, for each sir program, associates at each TEST of that program the LIST of LINES COVERED by it
    test_coverage_line_by_line = json_keys_to_int(json.load(file)) #{program1:{tc1:[linei,linej,...,linek],tc2:...}
with open("datasets/sir_programs/test_cases_costs.json", "r") as file:
    #dictionary that, for each sir program, associates at each TEST its EXECUTION COST
    test_cases_costs = json_keys_to_int(json.load(file)) #{program1:{tc1:ex_cost1,tc2:ex_cost2,...,tcn:ex_costn},program2:...}
with open("datasets/sir_programs/total_program_lines.json", "r") as file:
    #dictionary which associates at each SIR PROGRAM its size in terms of the NUMBER OF ITS LINES
    total_program_lines = json.load(file) #{program1:tot_lines_program1,program2:tot_lines_program2,program3:...}


def num_of_covered_lines(sir_program, test_cases):
    """This method returns the number of covered lines (no redundancy)"""
    covered_lines = set()

    for test_case in test_cases:
        try:
            for covered_line in test_coverage_line_by_line[sir_program][test_case]:
                covered_lines.add(covered_line)
        except:
            continue

    return len(covered_lines)


clusters_dictionary = dict()

for sir_program in sir_programs:
    tot_test_cases = sir_programs_tests_number[sir_program]

    # from {..., test_case_i : [cov_stmts], ...} to [..., #_stmt_cov_i, ...]
    test_cases_stmt_cov = []
    for test_case in test_coverage_line_by_line[sir_program].keys():
        test_cases_stmt_cov.append(len(test_coverage_line_by_line[sir_program][test_case]))
    suite_stmt_cov = sum(test_cases_stmt_cov)

    # Normalize data
    data = np.column_stack(
        (list(test_cases_costs[sir_program].values()), faults_dictionary[sir_program], test_cases_stmt_cov))
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    num_clusters = 50

    max_cluster_dim = 7

    # Step 2: Perform Hierarchical Clustering (Ward Method)
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

    clusters_dictionary[sir_program] = clustered_data

    # Step 3: Calculate the metrics for each cluster and validate
    cluster_metrics = {}
    for cluster_id in clustered_data.keys():
        tot_cluster_exec_cost = 0
        tot_cluster_past_fault_cov = 0
        tot_cluster_stmt_cov = 0
        for test_case in clustered_data[cluster_id]:
            tot_cluster_exec_cost += test_cases_costs[sir_program][test_case]
            tot_cluster_past_fault_cov += faults_dictionary[sir_program][test_case]
        tot_cluster_past_fault_cov = tot_cluster_past_fault_cov / tot_test_cases
        tot_cluster_stmt_cov = num_of_covered_lines(sir_program, clustered_data[cluster_id]) / total_program_lines[
            sir_program]
        cluster_metrics[cluster_id] = {
            "tot_exec_cost": tot_cluster_exec_cost,
            "tot_past_fault_cov": tot_cluster_past_fault_cov,
            "tot_stmt_cov": tot_cluster_stmt_cov  # Avg stmt coverage per test case in cluster
        }
        print(f"Cluster {cluster_id + 1} metrics:")
        print(f"Test Cases: {clustered_data[cluster_id]}")
        print(f" - Num. Test Cases: {len(clustered_data[cluster_id]):.2f}")
        print(f" - Execution Cost: {tot_cluster_exec_cost:.2f}")
        print(f" - Past Fault Coverage (%): {tot_cluster_past_fault_cov}")
        print(f" - Statement Coverage (%): {tot_cluster_stmt_cov:.2f}\n")

    for cluster_id in clustered_data.keys():
        print("Test cases of cluster " + str(cluster_id) + ": " + str(len(clustered_data[cluster_id])))

    print("======================================================================================")

    print("Program Name: " + sir_program)

    for cluster_id in clustered_data.keys():
        if len(clustered_data[cluster_id]) > max_cluster_dim:
            print("Test cases of cluster " + str(cluster_id) + ": " + str(len(clustered_data[cluster_id])))

    # Plotting the clusters in 3D space
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Extracting data for plotting
    exec_costs = np.array(list(test_cases_costs[sir_program].values()))
    fault_covs = np.array(faults_dictionary[sir_program])
    stmt_covs = np.array(test_cases_stmt_cov)

    # Plot each cluster with a different color
    colors = plt.cm.get_cmap("tab10", num_clusters)  # A colormap with as many colors as clusters
    for cluster_id in clustered_data.keys():
        cluster_indices = clustered_data[cluster_id]

        # Plot each cluster's points
        ax.scatter(
            exec_costs[cluster_indices],
            fault_covs[cluster_indices],
            stmt_covs[cluster_indices],
            color=colors(cluster_id),
            label=f"Cluster {cluster_id + 1}"
        )

    # Label the axes
    ax.set_xlabel("Execution Cost")
    ax.set_ylabel("Past Fault Coverage")
    ax.set_zlabel("Statement Coverage")
    ax.legend()
    ax.set_title("Test Case Clustering Visualization")

    # Display the plot
    #plt.show()

print(clusters_dictionary)


def make_linear_terms(sir_program, cluster_test_cases, alpha):
    """Making the linear terms of QUBO"""
    max_cost = max(test_cases_costs[sir_program].values())

    estimated_costs = []

    # linear coefficients, that are the diagonal of the matrix encoding the QUBO
    for test_case in cluster_test_cases:
        estimated_costs.append((alpha * (test_cases_costs[sir_program][test_case] / max_cost)) - (1 - alpha) *
                               faults_dictionary[sir_program][test_case])

    return np.array(estimated_costs)


def make_quadratic_terms(sir_program, variables, cluster_test_cases, linear_terms, penalty):
    """Making the quadratic terms of QUBO"""
    quadratic_terms = {}

    # k is a stmt
    for k in executed_lines_test_by_test[sir_program].keys():
        # k_test_cases is the list of test cases covering k
        k_test_cases = executed_lines_test_by_test[sir_program][k]
        for i in k_test_cases:
            if i not in cluster_test_cases or i not in variables:
                continue
            for j in k_test_cases:
                if j not in cluster_test_cases or j not in variables:
                    continue
                if i < j:
                    linear_terms[variables.index(i)] -= penalty
                    try:
                        quadratic_terms[variables.index(i), variables.index(j)] += 2 * penalty
                    except:
                        quadratic_terms[variables.index(i), variables.index(j)] = 2 * penalty

    return quadratic_terms


def create_QUBO_problem(linear_terms, quadratic_terms):
    """This function is the one that has to encode the QUBO problem that QAOA will have to solve. The QUBO problem specifies the optimization to solve and a quadratic binary unconstrained problem"""
    qubo = QuadraticProgram()

    for i in range(0, len(linear_terms)):
        qubo.binary_var('x%s' % (i))

    qubo.minimize(linear=linear_terms, quadratic=quadratic_terms)

    return qubo

penalties_dictionary = {"flex":None,"grep":None,"gzip":None,"sed":None}

#to get a QUBO problem from a quadratic problem with constraints, we have to insert those constraints into the Hamiltonian to solve (which is the one encoded by the QUBO problem). When we insert constraint into the Hamiltonian, we have to specify also penalties
for sir_program in sir_programs:
    max_penalty = 0
    max_cost = max(test_cases_costs[sir_program].values())
    for i in range(sir_programs_tests_number[sir_program]):
        cost = (alpha * (test_cases_costs[sir_program][i]/max_cost)) - ((1 - alpha) * faults_dictionary[sir_program][i])
        if cost > max_penalty:
            max_penalty = cost
    penalties_dictionary[sir_program] = max_penalty + 1

qubos_dictionary = {"flex":[],"grep":[],"gzip":[],"sed":[]}
converter = QuadraticProgramToQubo()
#make a dictionary that saves, for each program, the correspondent QUBO
for sir_program in sir_programs:
    print("SIR Program:\n")
    for cluster_id in clusters_dictionary[sir_program]:
        print("Cluster ID: " + str(cluster_id))
        variables = []
        for idx in range(0, len(clusters_dictionary[sir_program][cluster_id])):
            variables.append(idx)
        linear_terms = make_linear_terms(sir_program, clusters_dictionary[sir_program][cluster_id], alpha)
        quadratic_terms = make_quadratic_terms(sir_program, variables, clusters_dictionary[sir_program][cluster_id], linear_terms, penalties_dictionary[sir_program])
        qubo = create_QUBO_problem(linear_terms, quadratic_terms)
        qubos_dictionary[sir_program].append(qubo)
        print(qubo)
        print("/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/")
    print("======================================================================================")


def covered_lines(sir_program, test_cases_list):
    """Number of covered lines (no redundancy)"""
    covered_lines = set()

    for test_case in test_cases_list:
        try:
            for covered_line in test_coverage_line_by_line[sir_program][test_case]:
                covered_lines.add(covered_line)
        except:
            continue

    return len(covered_lines)


def build_pareto_front(sir_program, selected_tests):
    """This method builds the pareto front additionally from a sub test suite solution"""
    pareto_front = []
    max_fault_coverage = 0
    max_stmt_coverage = 0

    for index in range(1, len(selected_tests) + 1):
        # exract the first index selected tests
        candidate_solution = selected_tests[:index]
        candidate_solution_fault_coverage = 0
        candidate_solution_stmt_coverage = 0
        for selected_test in candidate_solution:
            candidate_solution_fault_coverage += faults_dictionary[sir_program][selected_test]
            candidate_solution_stmt_coverage += covered_lines(sir_program, candidate_solution)
        # if the actual pareto front dominates the candidate solution, get to the next candidate
        if max_fault_coverage >= candidate_solution_fault_coverage and max_stmt_coverage >= candidate_solution_stmt_coverage:
            continue
        # eventually update the pareto front information
        if candidate_solution_stmt_coverage > max_stmt_coverage:
            max_stmt_coverage = candidate_solution_stmt_coverage
        if candidate_solution_fault_coverage > max_fault_coverage:
            max_fault_coverage = candidate_solution_fault_coverage
        # add the candidate solution to the pareto front
        pareto_front.append(candidate_solution)

    return pareto_front


ideal_sampler = AerSampler()
ideal_sampler.options.shots = None

# I want to run the sampler 10 times to get different results for each sir program
for sir_program in sir_programs:
    for reps in [1]:
        qaoa = QAOA(sampler=ideal_sampler, optimizer=COBYLA(maxiter=500), reps=reps)
        # the fronts will be saved into files
        print("SIR Program: " + sir_program)
        file_path = "results/selectqaoa/statevector_sim/" + sir_program + "-data-rep-" + str(reps) + ".json"
        subsuites_file_path = "results/selectqaoa/aer_sim/" + sir_program + "-subsuites-data.json"
        subsuites_json_data = {}
        json_data = {}
        response = None
        qpu_run_times = []
        pareto_fronts_building_times = []
        for i in range(experiments):
            final_selected_tests = []
            experiment_clusters = []
            cluster_dict_index = 0
            for qubo in qubos_dictionary[sir_program]:
                print("QUBO Problem: " + str(qubo) + "\n Cluster Number: " + str(cluster_dict_index))
                print(
                    "Cluster's Test Cases: " + str(list(clusters_dictionary[sir_program].values())[cluster_dict_index]))
                # for each iteration get the result
                operator, offset = qubo.to_ising()
                print("Linear QUBO: " + str(qubo))
                # for each iteration get the result
                s = time.time()
                qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
                e = time.time()
                # print("QAOA Result: " + str(qaoa_result))
                qpu_run_times.append((e - s) * 1000)

                eigenstate = qaoa_result.eigenstate
                most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]

                # Convert to bitstring format
                if isinstance(most_likely, int):
                    n = qubo.get_num_binary_vars()
                    bitstring = [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
                elif isinstance(most_likely, str):
                    bitstring = [int(b) for b in most_likely[::-1]]
                else:
                    raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")

                f_value = qubo.objective.evaluate(bitstring)
                indexes_selected_tests = [index for index, value in enumerate(bitstring) if value == 1]
                print("Indexes of selected tests to convert. " + str(indexes_selected_tests))
                selected_tests = []
                for index in indexes_selected_tests:
                    selected_tests.append(list(clusters_dictionary[sir_program].values())[cluster_dict_index][index])
                experiment_clusters.append({
                    "cluster_number": cluster_dict_index,
                    "cluster_test_cases": list(clusters_dictionary[sir_program].values())[cluster_dict_index],
                    "bitstring": bitstring,
                    "f_value": f_value,
                    "indexes_selected_tests": indexes_selected_tests,
                    "selected_tests": selected_tests
                })
                print("Selected tests: " + str(selected_tests))
                print("Experiment Number: " + str(i))
                cluster_dict_index += 1
                for selected_test in selected_tests:
                    if selected_test not in final_selected_tests:
                        final_selected_tests.append(selected_test)
            subsuites_json_data["experiment_" + str(i)] = experiment_clusters
            i += 1
            # now we have to build the pareto front
            print("Final Selected Test Cases: " + str(final_selected_tests))
            print("Length of the final list of selected test cases: " + str(len(final_selected_tests)))
            start = time.time()
            pareto_front = build_pareto_front(sir_program, final_selected_tests)
            end = time.time()
            json_data["pareto_front_" + str(i)] = pareto_front
            pareto_front_building_time = (end - start) * 1000
            pareto_fronts_building_times.append(pareto_front_building_time)

        # compute the average time needed for the construction of a pareto frontier and run time
        mean_qpu_run_time = statistics.mean(qpu_run_times)
        mean_pareto_fronts_building_time = statistics.mean(pareto_fronts_building_times)
        json_data["mean_qpu_run_time(ms)"] = mean_qpu_run_time
        json_data["stdev_qpu_run_time(ms)"] = statistics.stdev(qpu_run_times)
        json_data["all_qpu_run_times(ms)"] = qpu_run_times
        json_data["mean_pareto_fronts_building_time(ms)"] = mean_pareto_fronts_building_time

        """with open(file_path, "w") as file:
            json.dump(json_data, file)"""

        with open(subsuites_file_path, "w") as file:
            json.dump(subsuites_json_data, file)

# I want to run the sampler 30 times to get different results for each sir program
sampling_noise_sampler = AerSampler()
sampling_noise_sampler.options.shots = 2048

for sir_program in sir_programs:
    qaoa = QAOA(sampler=sampling_noise_sampler, optimizer=COBYLA(500), reps=sir_programs_rep_values[sir_program])
    # the fronts will be saved into files
    print("SIR Program: " + sir_program)
    file_path = "results/selectqaoa/aer_sim/" + sir_program + "-data.json"
    subsuites_file_path = "results/selectqaoa/aer_sim/" + sir_program + "-subsuites-data.json"
    subsuites_json_data = {}
    json_data = {}
    response = None
    qpu_run_times = []
    pareto_fronts_building_times = []
    for i in range(experiments):
        final_selected_tests = []
        experiment_clusters = []
        cluster_dict_index = 0
        for qubo in qubos_dictionary[sir_program]:
            print("QUBO Problem: " + str(qubo) + "\n Cluster Number: " + str(cluster_dict_index))
            print("Cluster's Test Cases: " + str(list(clusters_dictionary[sir_program].values())[cluster_dict_index]))
            # for each iteration get the result
            operator, offset = qubo.to_ising()
            print("Linear QUBO: " + str(qubo))
            # for each iteration get the result
            s = time.time()
            qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
            e = time.time()
            # print("QAOA Result: " + str(qaoa_result))
            qpu_run_times.append((e - s) * 1000)

            eigenstate = qaoa_result.eigenstate
            most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]

            # Convert to bitstring format
            if isinstance(most_likely, int):
                n = qubo.get_num_binary_vars()
                bitstring = [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
            elif isinstance(most_likely, str):
                bitstring = [int(b) for b in most_likely[::-1]]
            else:
                raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")

            f_value = qubo.objective.evaluate(bitstring)
            print("f_value: " + str(f_value))
            indexes_selected_tests = [index for index, value in enumerate(bitstring) if value == 1]
            print("Indexes of selected tests to convert. " + str(indexes_selected_tests))
            selected_tests = []
            for index in indexes_selected_tests:
                selected_tests.append(list(clusters_dictionary[sir_program].values())[cluster_dict_index][index])
            experiment_clusters.append({
                "cluster_number": cluster_dict_index,
                "cluster_test_cases": list(clusters_dictionary[sir_program].values())[cluster_dict_index],
                "bitstring": bitstring,
                "f_value": f_value,
                "indexes_selected_tests": indexes_selected_tests,
                "selected_tests": selected_tests
            })
            print("Selected tests: " + str(selected_tests))
            print("Experiment Number: " + str(i))
            cluster_dict_index += 1
            for selected_test in selected_tests:
                if selected_test not in final_selected_tests:
                    final_selected_tests.append(selected_test)

        subsuites_json_data["experiment_" + str(i)] = experiment_clusters
        i += 1
        # now we have to build the pareto front
        print("Final Selected Test Cases: " + str(final_selected_tests))
        print("Length of the final list of selected test cases: " + str(len(final_selected_tests)))
        start = time.time()
        pareto_front = build_pareto_front(sir_program, final_selected_tests)
        end = time.time()
        json_data["pareto_front_" + str(i)] = pareto_front
        pareto_front_building_time = (end - start) * 1000
        pareto_fronts_building_times.append(pareto_front_building_time)

    # compute the average time needed for the construction of a pareto frontier and run time
    mean_qpu_run_time = statistics.mean(qpu_run_times)
    mean_pareto_fronts_building_time = statistics.mean(pareto_fronts_building_times)
    json_data["mean_qpu_run_time(ms)"] = mean_qpu_run_time
    json_data["stdev_qpu_run_time(ms)"] = statistics.stdev(qpu_run_times)
    json_data["all_qpu_run_times(ms)"] = qpu_run_times
    json_data["mean_pareto_fronts_building_time(ms)"] = mean_pareto_fronts_building_time

    """with open(file_path, "w") as file:
        json.dump(json_data, file)"""

    with open(subsuites_file_path, "w") as file:
        json.dump(subsuites_json_data, file)

noise_model = NoiseModel.from_backend(FakeBrisbane())
fake_sampler = AerSampler(backend_options={'noise_model': noise_model})
fake_sampler.options.shots = 2048

# I want to run the sampler 30 times to obtain different results for each sir program
for sir_program in sir_programs:
    qaoa = QAOA(sampler=fake_sampler, optimizer=COBYLA(500), reps=sir_programs_rep_values[sir_program])
    # the fronts will be saved into files
    print(sir_program)
    file_path = "results/selectqaoa/fake_brisbane/" + sir_program + "-data.json"
    subsuites_file_path = "results/selectqaoa/aer_sim/" + sir_program + "-subsuites-data.json"
    subsuites_json_data = {}
    json_data = {}
    response = None
    qpu_run_times = []
    pareto_fronts_building_times = []
    for i in range(experiments):
        final_selected_tests = []
        experiment_clusters = []
        cluster_dict_index = 0
        for qubo in qubos_dictionary[sir_program]:
            print("Experiment Number: " + str(i))
            print("QUBO Problem: " + str(qubo) + "\nNumber: " + str(cluster_dict_index))
            print("Cluster's Test Cases: " + str(list(clusters_dictionary[sir_program].values())[cluster_dict_index]))
            # for each iteration get the result
            operator, offset = qubo.to_ising()
            print("Linear QUBO: " + str(qubo))
            # for each iteration get the result
            s = time.time()
            qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
            e = time.time()
            # print("QAOA Result: " + str(qaoa_result))
            qpu_run_times.append((e - s) * 1000)

            eigenstate = qaoa_result.eigenstate
            most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]

            # Convert to bitstring format
            if isinstance(most_likely, int):
                n = qubo.get_num_binary_vars()
                bitstring = [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
            elif isinstance(most_likely, str):
                bitstring = [int(b) for b in most_likely[::-1]]
            else:
                raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")

            f_value = qubo.objective.evaluate(bitstring)
            indexes_selected_tests = [index for index, value in enumerate(bitstring) if value == 1]
            print("Indexes of selected tests to convert. " + str(indexes_selected_tests))
            selected_tests = []
            for index in indexes_selected_tests:
                selected_tests.append(list(clusters_dictionary[sir_program].values())[cluster_dict_index][index])
            experiment_clusters.append({
                "cluster_number": cluster_dict_index,
                "cluster_test_cases": list(clusters_dictionary[sir_program].values())[cluster_dict_index],
                "bitstring": bitstring,
                "f_value": f_value,
                "indexes_selected_tests": indexes_selected_tests,
                "selected_tests": selected_tests
            })
            print("Selected tests: " + str(selected_tests))
            cluster_dict_index += 1
            for selected_test in selected_tests:
                if selected_test not in final_selected_tests:
                    final_selected_tests.append(selected_test)
        subsuites_json_data["experiment_" + str(i)] = experiment_clusters
        i += 1
        # now we have to build the pareto front
        print("Final Selected Test Cases: " + str(final_selected_tests))
        print(len(final_selected_tests))
        start = time.time()
        pareto_front = build_pareto_front(sir_program, final_selected_tests)
        end = time.time()
        json_data["pareto_front_" + str(i)] = pareto_front
        pareto_front_building_time = (end - start) * 1000
        pareto_fronts_building_times.append(pareto_front_building_time)

    # compute the average time needed for the construction of a pareto frontier and run time
    mean_qpu_run_time = statistics.mean(qpu_run_times)
    mean_pareto_fronts_building_time = statistics.mean(pareto_fronts_building_times)
    json_data["mean_qpu_run_time(ms)"] = mean_qpu_run_time
    json_data["stdev_qpu_run_time(ms)"] = statistics.stdev(qpu_run_times)
    json_data["all_qpu_run_times(ms)"] = qpu_run_times
    json_data["mean_pareto_fronts_building_time(ms)"] = mean_pareto_fronts_building_time

    """with open(file_path, "w") as file:
        json.dump(json_data, file)"""

    with open(subsuites_file_path, "w") as file:
        json.dump(subsuites_json_data, file)

noise_model = NoiseModel()
error_1 = depolarizing_error(0.01, 1)
error_2 = depolarizing_error(0.01, 2)
noise_model.add_all_qubit_quantum_error(error_1, ['x', 'y', 'z', 'h', 'ry', 'rz', 'rx', 'sx', 'id'])
noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

noisy_sampler = AerSampler(backend_options={'noise_model': noise_model})
noisy_sampler.options.shots = 2048

# I want to run the sampler 30 times to obtain different results for each sir program
for sir_program in sir_programs:
    qaoa = QAOA(sampler=noisy_sampler, optimizer=COBYLA(500), reps=sir_programs_rep_values[sir_program])
    # the fronts will be saved into files
    print(sir_program)
    file_path = "results/selectqaoa/depolarizing_sim/01/" + sir_program + "-data.json"
    subsuites_file_path = "results/selectqaoa/aer_sim/" + sir_program + "-subsuites-data.json"
    subsuites_json_data = {}
    json_data = {}
    response = None
    qpu_run_times = []
    pareto_fronts_building_times = []
    for i in range(experiments):
        final_selected_tests = []
        experiment_clusters = []
        cluster_dict_index = 0
        for qubo in qubos_dictionary[sir_program]:
            print("Experiment Number: " + str(i))
            print("QUBO Problem: " + str(qubo) + "\nNumber: " + str(cluster_dict_index))
            print("Cluster's Test Cases: " + str(list(clusters_dictionary[sir_program].values())[cluster_dict_index]))
            # for each iteration get the result
            operator, offset = qubo.to_ising()
            print("Linear QUBO: " + str(qubo))
            # for each iteration get the result
            s = time.time()
            qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
            e = time.time()
            # print("QAOA Result: " + str(qaoa_result))
            qpu_run_times.append((e - s) * 1000)

            eigenstate = qaoa_result.eigenstate
            most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]

            # Convert to bitstring format
            if isinstance(most_likely, int):
                n = qubo.get_num_binary_vars()
                bitstring = [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
            elif isinstance(most_likely, str):
                bitstring = [int(b) for b in most_likely[::-1]]
            else:
                raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")

            f_value = qubo.objective.evaluate(bitstring)
            indexes_selected_tests = [index for index, value in enumerate(bitstring) if value == 1]
            print("Indexes of selected tests to convert. " + str(indexes_selected_tests))
            selected_tests = []
            for index in indexes_selected_tests:
                selected_tests.append(list(clusters_dictionary[sir_program].values())[cluster_dict_index][index])
            experiment_clusters.append({
                "cluster_number": cluster_dict_index,
                "cluster_test_cases": list(clusters_dictionary[sir_program].values())[cluster_dict_index],
                "bitstring": bitstring,
                "f_value": f_value,
                "indexes_selected_tests": indexes_selected_tests,
                "selected_tests": selected_tests
            })
            print("Selected tests: " + str(selected_tests))
            cluster_dict_index += 1
            for selected_test in selected_tests:
                if selected_test not in final_selected_tests:
                    final_selected_tests.append(selected_test)
        subsuites_json_data["experiment_" + str(i)] = experiment_clusters
        i += 1
        # now we have to build the pareto front
        print("Final Selected Test Cases: " + str(final_selected_tests))
        print(len(final_selected_tests))
        start = time.time()
        pareto_front = build_pareto_front(sir_program, final_selected_tests)
        end = time.time()
        json_data["pareto_front_" + str(i)] = pareto_front
        pareto_front_building_time = (end - start) * 1000
        pareto_fronts_building_times.append(pareto_front_building_time)

    # compute the average time needed for the construction of a pareto frontier and run time
    mean_qpu_run_time = statistics.mean(qpu_run_times)
    mean_pareto_fronts_building_time = statistics.mean(pareto_fronts_building_times)
    json_data["mean_qpu_run_time(ms)"] = mean_qpu_run_time
    json_data["stdev_qpu_run_time(ms)"] = statistics.stdev(qpu_run_times)
    json_data["all_qpu_run_times(ms)"] = qpu_run_times
    json_data["mean_pareto_fronts_building_time(ms)"] = mean_pareto_fronts_building_time

    """with open(file_path, "w") as file:
        json.dump(json_data, file)"""

    with open(subsuites_file_path, "w") as file:
        json.dump(subsuites_json_data, file)

noise_model = NoiseModel()
error_1 = depolarizing_error(0.02, 1)
error_2 = depolarizing_error(0.02, 2)
noise_model.add_all_qubit_quantum_error(error_1, ['x', 'y', 'z', 'h', 'ry', 'rz', 'rx', 'sx', 'id'])
noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

noisy_sampler = AerSampler(backend_options={'noise_model': noise_model})
noisy_sampler.options.shots = 2048

# I want to run the sampler 30 times to obtain different results for each sir program
for sir_program in sir_programs:
    qaoa = QAOA(sampler=noisy_sampler, optimizer=COBYLA(500), reps=sir_programs_rep_values[sir_program])
    # the fronts will be saved into files
    print(sir_program)
    file_path = "results/selectqaoa/depolarizing_sim/02/" + sir_program + "-data.json"
    subsuites_file_path = "results/selectqaoa/aer_sim/" + sir_program + "-subsuites-data.json"
    subsuites_json_data = {}
    json_data = {}
    response = None
    qpu_run_times = []
    pareto_fronts_building_times = []
    for i in range(experiments):
        final_selected_tests = []
        experiment_clusters = []
        cluster_dict_index = 0
        for qubo in qubos_dictionary[sir_program]:
            print("Experiment Number: " + str(i))
            print("QUBO Problem: " + str(qubo) + "\nNumber: " + str(cluster_dict_index))
            print("Cluster's Test Cases: " + str(list(clusters_dictionary[sir_program].values())[cluster_dict_index]))
            # for each iteration get the result
            operator, offset = qubo.to_ising()
            print("Linear QUBO: " + str(qubo))
            # for each iteration get the result
            s = time.time()
            qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
            e = time.time()
            # print("QAOA Result: " + str(qaoa_result))
            qpu_run_times.append((e - s) * 1000)

            eigenstate = qaoa_result.eigenstate
            most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]

            # Convert to bitstring format
            if isinstance(most_likely, int):
                n = qubo.get_num_binary_vars()
                bitstring = [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
            elif isinstance(most_likely, str):
                bitstring = [int(b) for b in most_likely[::-1]]
            else:
                raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")

            f_value = qubo.objective.evaluate(bitstring)
            indexes_selected_tests = [index for index, value in enumerate(bitstring) if value == 1]
            print("Indexes of selected tests to convert. " + str(indexes_selected_tests))
            selected_tests = []
            for index in indexes_selected_tests:
                selected_tests.append(list(clusters_dictionary[sir_program].values())[cluster_dict_index][index])
            experiment_clusters.append({
                "cluster_number": cluster_dict_index,
                "cluster_test_cases": list(clusters_dictionary[sir_program].values())[cluster_dict_index],
                "bitstring": bitstring,
                "f_value": f_value,
                "indexes_selected_tests": indexes_selected_tests,
                "selected_tests": selected_tests
            })
            print("Selected tests: " + str(selected_tests))
            cluster_dict_index += 1
            for selected_test in selected_tests:
                if selected_test not in final_selected_tests:
                    final_selected_tests.append(selected_test)
        subsuites_json_data["experiment_" + str(i)] = experiment_clusters
        i += 1
        # now we have to build the pareto front
        print("Final Selected Test Cases: " + str(final_selected_tests))
        print(len(final_selected_tests))
        start = time.time()
        pareto_front = build_pareto_front(sir_program, final_selected_tests)
        end = time.time()
        json_data["pareto_front_" + str(i)] = pareto_front
        pareto_front_building_time = (end - start) * 1000
        pareto_fronts_building_times.append(pareto_front_building_time)

    # compute the average time needed for the construction of a pareto frontier and run time
    mean_qpu_run_time = statistics.mean(qpu_run_times)
    mean_pareto_fronts_building_time = statistics.mean(pareto_fronts_building_times)
    json_data["mean_qpu_run_time(ms)"] = mean_qpu_run_time
    json_data["stdev_qpu_run_time(ms)"] = statistics.stdev(qpu_run_times)
    json_data["all_qpu_run_times(ms)"] = qpu_run_times
    json_data["mean_pareto_fronts_building_time(ms)"] = mean_pareto_fronts_building_time

    """with open(file_path, "w") as file:
        json.dump(json_data, file)"""

    with open(subsuites_file_path, "w") as file:
        json.dump(subsuites_json_data, file)

noise_model = NoiseModel()
error_1 = depolarizing_error(0.05, 1)
error_2 = depolarizing_error(0.05, 2)
noise_model.add_all_qubit_quantum_error(error_1, ['x', 'y', 'z', 'h', 'ry', 'rz', 'rx', 'sx', 'id'])
noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

noisy_sampler = AerSampler(backend_options={'noise_model': noise_model})
noisy_sampler.options.shots = 2048

# I want to run the sampler 30 times to obtain different results for each sir program
for sir_program in sir_programs:
    qaoa = QAOA(sampler=noisy_sampler, optimizer=COBYLA(500), reps=sir_programs_rep_values[sir_program])
    # the fronts will be saved into files
    print(sir_program)
    file_path = "results/selectqaoa/depolarizing_sim/05/" + sir_program + "-data.json"
    subsuites_file_path = "results/selectqaoa/aer_sim/" + sir_program + "-subsuites-data.json"
    subsuites_json_data = {}
    json_data = {}
    response = None
    qpu_run_times = []
    pareto_fronts_building_times = []
    for i in range(experiments):
        final_selected_tests = []
        experiment_clusters = []
        cluster_dict_index = 0
        for qubo in qubos_dictionary[sir_program]:
            print("Experiment Number: " + str(i))
            print("QUBO Problem: " + str(qubo) + "\nNumber: " + str(cluster_dict_index))
            print("Cluster's Test Cases: " + str(list(clusters_dictionary[sir_program].values())[cluster_dict_index]))
            # for each iteration get the result
            operator, offset = qubo.to_ising()
            print("Linear QUBO: " + str(qubo))
            # for each iteration get the result
            s = time.time()
            qaoa_result = qaoa.compute_minimum_eigenvalue(operator)
            e = time.time()
            # print("QAOA Result: " + str(qaoa_result))
            qpu_run_times.append((e - s) * 1000)

            eigenstate = qaoa_result.eigenstate
            most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]

            # Convert to bitstring format
            if isinstance(most_likely, int):
                n = qubo.get_num_binary_vars()
                bitstring = [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
            elif isinstance(most_likely, str):
                bitstring = [int(b) for b in most_likely[::-1]]
            else:
                raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")

            f_value = qubo.objective.evaluate(bitstring)
            indexes_selected_tests = [index for index, value in enumerate(bitstring) if value == 1]
            print("Indexes of selected tests to convert. " + str(indexes_selected_tests))
            selected_tests = []
            for index in indexes_selected_tests:
                selected_tests.append(list(clusters_dictionary[sir_program].values())[cluster_dict_index][index])
            experiment_clusters.append({
                "cluster_number": cluster_dict_index,
                "cluster_test_cases": list(clusters_dictionary[sir_program].values())[cluster_dict_index],
                "bitstring": bitstring,
                "f_value": f_value,
                "indexes_selected_tests": indexes_selected_tests,
                "selected_tests": selected_tests
            })
            print("Selected tests: " + str(selected_tests))
            cluster_dict_index += 1
            for selected_test in selected_tests:
                if selected_test not in final_selected_tests:
                    final_selected_tests.append(selected_test)
        subsuites_json_data["experiment_" + str(i)] = experiment_clusters
        i += 1
        # now we have to build the pareto front
        print("Final Selected Test Cases: " + str(final_selected_tests))
        print(len(final_selected_tests))
        start = time.time()
        pareto_front = build_pareto_front(sir_program, final_selected_tests)
        end = time.time()
        json_data["pareto_front_" + str(i)] = pareto_front
        pareto_front_building_time = (end - start) * 1000
        pareto_fronts_building_times.append(pareto_front_building_time)

    # compute the average time needed for the construction of a pareto frontier and run time
    mean_qpu_run_time = statistics.mean(qpu_run_times)
    mean_pareto_fronts_building_time = statistics.mean(pareto_fronts_building_times)
    json_data["mean_qpu_run_time(ms)"] = mean_qpu_run_time
    json_data["stdev_qpu_run_time(ms)"] = statistics.stdev(qpu_run_times)
    json_data["all_qpu_run_times(ms)"] = qpu_run_times
    json_data["mean_pareto_fronts_building_time(ms)"] = mean_pareto_fronts_building_time

    """with open(file_path, "w") as file:
        json.dump(json_data, file)"""

    with open(subsuites_file_path, "w") as file:
        json.dump(subsuites_json_data, file)

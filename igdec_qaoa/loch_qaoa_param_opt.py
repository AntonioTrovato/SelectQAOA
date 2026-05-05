import optuna
import json
import os
import random
import time
import numpy as np
import pandas as pd
from typing import List, Union
from docplex.mp.model import Model
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.applications import OptimizationApplication

# ── Shared QAOA runner ────────────────────────────────────────────────────────

def run_alg(qubo, reps=1):
    ideal_sampler = AerSampler()
    ideal_sampler.options.shots = None
    qaoa = QAOA(sampler=ideal_sampler, optimizer=COBYLA(500), reps=reps)
    operator, _ = qubo.to_ising()
    s = time.time()
    result = qaoa.compute_minimum_eigenvalue(operator)
    e = time.time()
    return result, (e - s) * 1000

def extract_bitstring(result, qubo):
    eigenstate = result.eigenstate
    most_likely = max(eigenstate.items(), key=lambda x: x[1])[0]
    n = qubo.get_num_binary_vars()
    if isinstance(most_likely, int):
        return [int(b) for b in format(most_likely, f'0{n}b')[::-1]]
    elif isinstance(most_likely, str):
        return [int(b) for b in most_likely[::-1]]
    raise ValueError(f"Unsupported eigenstate key type: {type(most_likely)}")

# ── QUBO classes ──────────────────────────────────────────────────────────────

class TCO_TCM(OptimizationApplication):
    """3-objective QUBO: time, failure rate, count  (gsdtsr / iofrol / paintcontrol)"""
    def __init__(self, times, frs, w1, w2, w3, sample, solution):
        self._times, self._frs = times, frs
        self._w1, self._w2, self._w3 = w1, w2, w3
        self._sample, self._solution = sample, solution

    def to_quadratic_program(self):
        mdl = Model(name="TCM")
        x = {i: mdl.binary_var(name=f"x_{i}") for i in self._sample}
        obj_time = obj_rate = obj_num = 0
        for i in range(len(self._solution)):
            v = x[i] if i in self._sample else self._solution[i]
            obj_time += self._times[i] * v
            obj_rate += self._frs[i] * v
            obj_num  += v
        time_sum = sum(self._times)
        rate_sum = sum(self._frs)
        obj_time = pow(obj_time / time_sum, 2)
        obj_rate = pow((obj_rate - rate_sum) / rate_sum, 2)
        obj_num  = pow(obj_num / len(self._times), 2)
        mdl.minimize(self._w1 * obj_time + self._w2 * obj_rate + self._w3 * obj_num)
        return from_docplex_mp(mdl)

    def interpret(self, result):
        return [i for i, v in enumerate(self._result_to_x(result)) if v]


class TCO_ElevTwo(OptimizationApplication):
    """2-objective QUBO: cost, input_div  (elevator)"""
    def __init__(self, cost, div, w1, w2, sample, solution):
        self._cost, self._div = cost, div
        self._w1, self._w2 = w1, w2
        self._sample, self._solution = sample, solution

    def to_quadratic_program(self):
        mdl = Model(name="ElevTwo")
        x = {i: mdl.binary_var(name=f"x_{i}") for i in self._sample}
        obj_cost = obj_div = 0
        for i in range(len(self._solution)):
            v = x[i] if i in self._sample else self._solution[i]
            obj_cost += self._cost[i] * v
            obj_div  += self._div[i] * v
        cost_sum = sum(self._cost)
        div_sum  = sum(self._div)
        obj_cost = pow(obj_cost / cost_sum, 2)
        obj_div  = pow((obj_div - div_sum) / div_sum, 2)
        mdl.minimize(self._w1 * obj_cost + self._w2 * obj_div)
        return from_docplex_mp(mdl)

    def interpret(self, result):
        return [i for i, v in enumerate(self._result_to_x(result)) if v]


class TCO_ElevThree(OptimizationApplication):
    """3-objective QUBO: cost, pcount, dist  (elevator2)"""
    def __init__(self, cost, pcount, dist, w1, w2, w3, sample, solution):
        self._cost, self._pcount, self._dist = cost, pcount, dist
        self._w1, self._w2, self._w3 = w1, w2, w3
        self._sample, self._solution = sample, solution

    def to_quadratic_program(self):
        mdl = Model(name="ElevThree")
        x = {i: mdl.binary_var(name=f"x_{i}") for i in self._sample}
        obj_cost = obj_pcount = obj_dist = 0
        for i in range(len(self._solution)):
            v = x[i] if i in self._sample else self._solution[i]
            obj_cost   += self._cost[i]   * v
            obj_pcount += self._pcount[i] * v
            obj_dist   += self._dist[i]   * v
        cost_sum   = sum(self._cost)
        pcount_sum = sum(self._pcount)
        dist_sum   = sum(self._dist)
        obj_cost   = pow(obj_cost   / cost_sum,   2)
        obj_pcount = pow((obj_pcount - pcount_sum) / pcount_sum, 2)
        obj_dist   = pow((obj_dist   - dist_sum)   / dist_sum,   2)
        mdl.minimize(self._w1 * obj_cost + self._w2 * obj_pcount + self._w3 * obj_dist)
        return from_docplex_mp(mdl)

    def interpret(self, result):
        return [i for i, v in enumerate(self._result_to_x(result)) if v]

# ── Data loaders ──────────────────────────────────────────────────────────────

def load_data(program_name, dataset_dir="../datasets/quantum_sota_datasets"):
    path = os.path.join(dataset_dir, "elevator.csv" if "elevator" in program_name else f"{program_name}.csv")
    if program_name == "elevator":
        df = pd.read_csv(path, dtype={"cost": int, "input_div": float})
    elif program_name == "elevator2":
        df = pd.read_csv(path, dtype={"cost": int, "pcount": int, "dist": int})
    else:
        df = pd.read_csv(path, dtype={"time": float, "rate": float})
        df = df[df["rate"] > 0]
    return df

# ── fval helpers (mirror print_diet from each script) ─────────────────────────

def fval_tcm(solution, df):
    """Fitness for gsdtsr/iofrol/paintcontrol — matches loch_qaoa_tcm.print_diet"""
    sel = [i for i, v in enumerate(solution) if v == 1]
    times = [df.iloc[i]["time"] for i in sel]
    rates = [df.iloc[i]["rate"] for i in sel]
    t_sum = df["time"].sum(); r_sum = df["rate"].sum()
    return (
        (1/3) * pow(sum(times) / t_sum, 2)
        + (1/3) * pow((sum(rates) - r_sum + 1e-20) / (r_sum + 1e-20), 2)
        + (1/3) * pow(len(sel) / len(df), 2)
    )

def fval_elev_two(solution, df):
    """Fitness for elevator — matches loch_qaoa_elev_two.print_diet"""
    sel = [i for i, v in enumerate(solution) if v == 1]
    costs = [df.iloc[i]["cost"]      for i in sel]
    divs  = [df.iloc[i]["input_div"] for i in sel]
    c_sum = df["cost"].sum(); d_sum = df["input_div"].sum()
    return (
        (1/2) * pow(sum(costs) / c_sum, 2)
        + (1/2) * pow((sum(divs) - d_sum) / d_sum, 2)
    )

def fval_elev_three(solution, df):
    """Fitness for elevator2 — matches loch_qaoa_elev_three.print_diet"""
    sel = [i for i, v in enumerate(solution) if v == 1]
    costs   = [df.iloc[i]["cost"]   for i in sel]
    pcounts = [df.iloc[i]["pcount"] for i in sel]
    dists   = [df.iloc[i]["dist"]   for i in sel]
    c_sum = df["cost"].sum(); p_sum = df["pcount"].sum(); d_sum = df["dist"].sum()
    return (
        (1/3) * pow(sum(costs)   / c_sum, 2)
        + (1/3) * pow((sum(pcounts) - p_sum) / p_sum, 2)
        + (1/3) * pow((sum(dists)   - d_sum) / d_sum, 2)
    )

# ── Impact ordering (vectorised, mirrors OrderByImpactNum) ────────────────────

def impact_order_tcm(solution, df):
    times = list(df["time"]); rates = list(df["rate"])
    n = len(solution)
    matrix = np.array([solution] * n, dtype=float)
    for i in range(n):
        matrix[i][i] = 1 - matrix[i][i]
    t_sum = sum(times); r_sum = sum(rates)
    t_obj = matrix.dot(np.array(times).reshape(-1, 1))
    r_obj = matrix.dot(np.array(rates).reshape(-1, 1)) - r_sum
    num_obj = matrix.sum(axis=1, keepdims=True) / n
    energy = fval_tcm(solution, df)
    obj = (1/3)*(t_obj/t_sum)**2 + (1/3)*((r_obj+1e-20)/(r_sum+1e-20))**2 + (1/3)*num_obj**2 - energy
    return [i for i, _ in sorted(enumerate(obj.flatten()), key=lambda x: x[1])]

def impact_order_elev_two(solution, df):
    costs = list(df["cost"]); divs = list(df["input_div"])
    n = len(solution)
    matrix = np.array([solution] * n, dtype=float)
    for i in range(n):
        matrix[i][i] = 1 - matrix[i][i]
    c_sum = sum(costs); d_sum = sum(divs)
    c_obj = matrix.dot(np.array(costs).reshape(-1, 1))
    d_obj = matrix.dot(np.array(divs).reshape(-1, 1)) - d_sum
    energy = fval_elev_two(solution, df)
    obj = (1/2)*(c_obj/c_sum)**2 + (1/2)*(d_obj/d_sum)**2 - energy
    return [i for i, _ in sorted(enumerate(obj.flatten()), key=lambda x: x[1])]

def impact_order_elev_three(solution, df):
    costs = list(df["cost"]); pcounts = list(df["pcount"]); dists = list(df["dist"])
    n = len(solution)
    matrix = np.array([solution] * n, dtype=float)
    for i in range(n):
        matrix[i][i] = 1 - matrix[i][i]
    c_sum = sum(costs); p_sum = sum(pcounts); d_sum = sum(dists)
    c_obj = matrix.dot(np.array(costs).reshape(-1, 1))
    p_obj = matrix.dot(np.array(pcounts).reshape(-1, 1)) - p_sum
    d_obj = matrix.dot(np.array(dists).reshape(-1, 1))   - d_sum
    energy = fval_elev_three(solution, df)
    obj = (1/3)*(c_obj/c_sum)**2 + (1/3)*(p_obj/p_sum)**2 + (1/3)*(d_obj/d_sum)**2 - energy
    return [i for i, _ in sorted(enumerate(obj.flatten()), key=lambda x: x[1])]

# ── Core IGDec-QAOA loop (program-agnostic) ───────────────────────────────────

def run_igdec(program_name, weights, df, reps=1, max_iter=3, problem_size=7):
    """
    Runs one full IGDec-QAOA experiment and returns best_energy (lower = better).
    max_iter kept small (5 vs original 10) for Optuna speed; raise for final eval.
    """
    # --- resolve per-program helpers ---
    if program_name in ("gsdtsr", "iofrol", "paintcontrol"):
        times = df["time"].tolist(); rates = df["rate"].tolist()
        w1, w2, w3 = weights

        def make_qubo(case_list, solution):
            tc = TCO_TCM(times, rates, w1, w2, w3, case_list, solution)
            return tc.to_quadratic_program()

        def fval_fn(sol): return fval_tcm(sol, df)
        def order_fn(sol, energy): return impact_order_tcm(sol, df)

    elif program_name == "elevator":
        costs = df["cost"].tolist(); divs = df["input_div"].tolist()
        w1, w2 = weights

        def make_qubo(case_list, solution):
            tc = TCO_ElevTwo(costs, divs, w1, w2, case_list, solution)
            return tc.to_quadratic_program()

        def fval_fn(sol): return fval_elev_two(sol, df)
        def order_fn(sol, energy): return impact_order_elev_two(sol, df)

    else:  # elevator2
        costs = df["cost"].tolist(); pcounts = df["pcount"].tolist(); dists = df["dist"].tolist()
        w1, w2, w3 = weights

        def make_qubo(case_list, solution):
            tc = TCO_ElevThree(costs, pcounts, dists, w1, w2, w3, case_list, solution)
            return tc.to_quadratic_program()

        def fval_fn(sol): return fval_elev_three(sol, df)
        def order_fn(sol, energy): return impact_order_elev_three(sol, df)

    # --- initialise ---
    length = len(df)
    solution = [random.choice([0, 1]) for _ in range(length)]
    best_energy = fval_fn(solution)
    best_solution = solution.copy()
    imp_order = order_fn(solution, best_energy)

    index_begin, index_end = 0, problem_size
    count = 0

    while count < max_iter:
        if problem_size > 0.15 * length:
            # single sub-problem per iteration
            case_list = imp_order[index_begin:index_end]
            qubo = make_qubo(case_list, solution)
            result, _ = run_alg(qubo, reps)
            bitstring = extract_bitstring(result, qubo)
            for ci, case in enumerate(case_list):
                solution[case] = bitstring[ci]
            result_fval = qubo.objective.evaluate(bitstring)
        else:
            # slide window across impact order
            result_fval = best_energy
            while index_end <= 0.15 * length:
                case_list = imp_order[index_begin:index_end]
                qubo = make_qubo(case_list, solution)
                result, _ = run_alg(qubo, reps)
                bitstring = extract_bitstring(result, qubo)
                for ci, case in enumerate(case_list):
                    solution[case] = bitstring[ci]
                result_fval = qubo.objective.evaluate(bitstring)
                index_begin += problem_size
                index_end   += problem_size

        energy = result_fval
        if energy < best_energy:
            best_energy   = energy
            best_solution = solution.copy()

        imp_order = order_fn(solution, energy)
        index_begin, index_end = 0, problem_size
        count += 1

    return best_energy

# ── Optuna objectives ─────────────────────────────────────────────────────────

def make_objective(program_name, df, reps=1, n_experiments=3):
    def objective(trial):
        if program_name == "elevator":
            # 2 weights, normalised to sum = 1
            w1_raw = trial.suggest_float("w1", 0.0, 1.0)
            w2_raw = trial.suggest_float("w2", 0.0, 1.0)
            total  = (w1_raw + w2_raw) or 1.0
            weights = (w1_raw / total, w2_raw / total)
        else:
            # 3 weights (TCM and elevator2), normalised to sum = 1
            w1_raw = trial.suggest_float("w1", 0.0, 1.0)
            w2_raw = trial.suggest_float("w2", 0.0, 1.0)
            w3_raw = trial.suggest_float("w3", 0.0, 1.0)
            total  = (w1_raw + w2_raw + w3_raw) or 1.0
            weights = (w1_raw / total, w2_raw / total, w3_raw / total)

        scores = [
            run_igdec(program_name, weights, df, reps=reps)
            for _ in range(n_experiments)
        ]
        return float(np.mean(scores))

    return objective

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    programs = ["gsdtsr", "iofrol", "paintcontrol", "elevator", "elevator2"]
    reps = 1
    n_trials = 10
    n_experiments = 3   # increase to 10 for final eval

    output_dir = "../results/igdec_qaoa/ideal"
    os.makedirs(output_dir, exist_ok=True)

    opt_params = {}

    for program in programs:
        print(f"\n{'='*60}\nOptimizing weights for: {program}\n{'='*60}")
        df = load_data(program)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
            study_name=f"igdec_{program}",
        )
        study.optimize(
            make_objective(program, df, reps=reps, n_experiments=n_experiments),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best = study.best_trial
        opt_params[program] = best.params
        print(f"Best params : {best.params}")
        print(f"Best score  : {best.value:.6f}")

    # Save JSON
    out_path = os.path.join(output_dir, "so_opt_params.json")
    with open(out_path, "w") as f:
        json.dump(opt_params, f, indent=2)
    print(f"\nOptimal parameters saved to {out_path}")
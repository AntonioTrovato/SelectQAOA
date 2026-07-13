"""
Cluster-level comparison of the 6 QAOA simulators.

For each program, each repetition (experiment), and each cluster, this script
looks at which simulator found the lowest f_value (i.e. the best QUBO
solution) for that exact cluster, and tallies a "win" for it. Since the
clustering is computed only once and reused by all 6 simulators, clusters are
matched positionally (same index = same cluster) across the subsuites files.

Outputs (in OUTPUT_DIR):
  - cluster_level_comparison.csv : one row per (program, experiment, cluster,
    simulator) with the f_value and whether it won that cluster
  - wins_per_experiment.csv      : win_count per (program, experiment, simulator)
  - program_summary.csv          : mean/std/sum of win_count per (program, simulator)
  - <program>_boxplot.png        : distribution (over the 10 repetitions) of
    clusters won, one box per simulator
  - aggregate_mean_wins.png      : bar chart, average clusters won per
    simulator, one group per program

Adjust SIMULATORS / SIR_PROGRAMS / TIE_HANDLING below to match your setup.
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG - adjust to match your actual file layout / program list
# ---------------------------------------------------------------------------

SIMULATORS = {
    "ideal":            "results/selectqaoa/statevector_sim/{program}-subsuites-data.json",
    "sampling_noise":   "results/selectqaoa/aer_sim/{program}-subsuites-data.json",
    "fake_brisbane":    "results/selectqaoa/fake_brisbane/{program}-subsuites-data.json",
    "depolarizing_01":  "results/selectqaoa/depolarizing_sim/01/{program}-subsuites-data.json",
    "depolarizing_02":  "results/selectqaoa/depolarizing_sim/02/{program}-subsuites-data.json",
    "depolarizing_05":  "results/selectqaoa/depolarizing_sim/05/{program}-subsuites-data.json",
}

SIR_PROGRAMS = ["flex", "grep", "gzip", "sed"]

# "all"  : every tied simulator gets a full +1 point
# "split": a tie for the best f_value splits 1 point across the tied sims
TIE_HANDLING = "all"

OUTPUT_DIR = "results/selectqaoa/cluster_level_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all_data():
    data = {}
    for sim_name, path_template in SIMULATORS.items():
        data[sim_name] = {}
        for program in SIR_PROGRAMS:
            path = path_template.format(program=program)
            if not os.path.exists(path):
                print(f"[WARNING] missing file, skipping: {path}")
                continue
            with open(path, "r") as f:
                data[sim_name][program] = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare(data):
    wins_long = []
    # counts_per_exp[program][experiment_key][simulator] = win score (float)
    counts_per_exp = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for program in SIR_PROGRAMS:
        exp_key_sets = [
            set(data[sim][program].keys())
            for sim in SIMULATORS
            if program in data[sim]
        ]
        if not exp_key_sets:
            continue
        common_exp_keys = sorted(
            set.intersection(*exp_key_sets),
            key=lambda k: int(k.split("_")[1]),
        )

        for exp_key in common_exp_keys:
            # Index each simulator's clusters by cluster_number (not list
            # position), so we're robust to any reordering between runs.
            per_sim_by_cluster_num = {}
            for sim_name in SIMULATORS:
                if program not in data[sim_name]:
                    continue
                entries = data[sim_name][program][exp_key]
                by_num = {}
                for entry in entries:
                    cnum = entry["cluster_number"]
                    if cnum in by_num:
                        print(
                            f"[WARNING] {program} {exp_key} {sim_name}: "
                            f"duplicate cluster_number={cnum}, keeping first"
                        )
                        continue
                    by_num[cnum] = entry
                per_sim_by_cluster_num[sim_name] = by_num

            # cluster_numbers must be present in every simulator to be compared
            cluster_num_sets = [set(d.keys()) for d in per_sim_by_cluster_num.values()]
            common_cluster_nums = sorted(set.intersection(*cluster_num_sets))
            all_cluster_nums = sorted(set.union(*cluster_num_sets))
            if len(common_cluster_nums) != len(all_cluster_nums):
                missing = set(all_cluster_nums) - set(common_cluster_nums)
                print(
                    f"[WARNING] {program} {exp_key}: cluster_number(s) {sorted(missing)} "
                    f"not present in all simulators, skipping those"
                )

            for cluster_num in common_cluster_nums:
                # sanity check: same cluster_number should mean the same set
                # of test cases across simulators (they share the same
                # upfront clustering). Warn (don't crash) if it doesn't.
                ref_sim_name = next(iter(per_sim_by_cluster_num))
                ref_entry = per_sim_by_cluster_num[ref_sim_name][cluster_num]
                ref_test_cases = set(ref_entry.get("cluster_test_cases", []))
                for sim_name, by_num in per_sim_by_cluster_num.items():
                    tc = set(by_num[cluster_num].get("cluster_test_cases", []))
                    if tc != ref_test_cases:
                        print(
                            f"[WARNING] {program} {exp_key} cluster_number={cluster_num}: "
                            f"cluster_test_cases differ for simulator '{sim_name}' "
                            f"vs reference — check your saved data"
                        )

                f_values = {
                    sim_name: by_num[cluster_num]["f_value"]
                    for sim_name, by_num in per_sim_by_cluster_num.items()
                }

                best_f = min(f_values.values())
                winners = [s for s, v in f_values.items() if v == best_f]
                score_per_winner = 1.0 / len(winners) if TIE_HANDLING == "split" else 1.0

                for sim_name, f_val in f_values.items():
                    is_winner = sim_name in winners
                    win_score = score_per_winner if is_winner else 0.0
                    wins_long.append({
                        "program": program,
                        "experiment": exp_key,
                        "cluster_number": cluster_num,
                        "simulator": sim_name,
                        "f_value": f_val,
                        "is_winner": is_winner,
                        "win_score": win_score,
                    })
                    counts_per_exp[program][exp_key][sim_name] += win_score

    return wins_long, counts_per_exp


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def build_summary_tables(counts_per_exp):
    rows = []
    for program, exp_dict in counts_per_exp.items():
        for exp_key, sim_dict in exp_dict.items():
            for sim_name, score in sim_dict.items():
                rows.append({
                    "program": program,
                    "experiment": exp_key,
                    "simulator": sim_name,
                    "win_count": score,
                })
    df_per_exp = pd.DataFrame(rows)

    df_program_summary = (
        df_per_exp.groupby(["program", "simulator"])["win_count"]
        .agg(["mean", "std", "sum"])
        .reset_index()
        .rename(columns={
            "mean": "mean_win_count",
            "std": "std_win_count",
            "sum": "total_win_count",
        })
    )
    return df_per_exp, df_program_summary


def print_best_per_program_experiment(df_per_exp):
    print("\n=== Best simulator per program & experiment ===")
    for (program, exp_key), group in df_per_exp.groupby(["program", "experiment"]):
        top = group.loc[group["win_count"].idxmax()]
        print(
            f"{program:8s} {exp_key:14s} -> best: {top['simulator']:16s} "
            f"(won {top['win_count']:.1f} clusters)"
        )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_boxplots(df_per_exp):
    for program in df_per_exp["program"].unique():
        sub = df_per_exp[df_per_exp["program"] == program]
        pivot = sub.pivot(index="experiment", columns="simulator", values="win_count")
        pivot = pivot[[s for s in SIMULATORS if s in pivot.columns]]

        plt.figure(figsize=(8, 5))
        pivot.boxplot()
        plt.title(f"Cluster-level wins per simulator — {program}")
        plt.ylabel("Number of clusters won (per repetition)")
        plt.xlabel("Simulator")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"{program}_boxplot.pdf")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved boxplot: {out_path}")


def plot_aggregate_score(df_program_summary):
    pivot = df_program_summary.pivot(index="program", columns="simulator", values="mean_win_count")
    pivot = pivot[[s for s in SIMULATORS if s in pivot.columns]]

    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("Mean # of clusters won per repetition")
    ax.set_title("Aggregate: average clusters won per simulator, per program")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "aggregate_mean_wins.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved aggregate plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = load_all_data()
    wins_long, counts_per_exp = compare(data)

    df_wins_long = pd.DataFrame(wins_long)
    df_wins_long.to_csv(os.path.join(OUTPUT_DIR, "cluster_level_comparison.csv"), index=False)

    df_per_exp, df_program_summary = build_summary_tables(counts_per_exp)
    df_per_exp.to_csv(os.path.join(OUTPUT_DIR, "wins_per_experiment.csv"), index=False)
    df_program_summary.to_csv(os.path.join(OUTPUT_DIR, "program_summary.csv"), index=False)

    print("\n=== Program-level summary (mean/std/sum of clusters won across repetitions) ===")
    print(df_program_summary.to_string(index=False))

    print_best_per_program_experiment(df_per_exp)

    overall = (
        df_program_summary.groupby("simulator")["mean_win_count"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\n=== Overall ranking (average across programs of mean wins per repetition) ===")
    print(overall.to_string())

    plot_boxplots(df_per_exp)
    plot_aggregate_score(df_program_summary)


if __name__ == "__main__":
    main()
"""Turn runs/*/group_stats.jsonl + eval.json into the paper's figures and tables.

Figures
    fig1_wasted_compute.png   frac of rollouts with zero gradient over training,
                              one curve per group size G          (twist #2)
    fig2_group_size.png       final accuracy vs G at matched rollout budget,
                              mean +/- range across seeds          (twists #1, #6)
    fig3_grader_table.md      accuracy by grader x run: does the ranking flip?
                              (twist #4)

Usage:
    python analyze.py                 # reads runs/, writes figures/
    python analyze.py --runs_dir runs --out_dir figures
    python analyze.py --selftest
"""

import argparse
import json
import os
from collections import defaultdict


def load_runs(runs_dir):
    """One record per run: its config, its group-stats timeseries, its eval results."""
    runs = []
    if not os.path.isdir(runs_dir):
        return runs
    for name in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, name)
        cfg_path = os.path.join(d, "config.json")
        if not os.path.isfile(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        stats = []
        sp = os.path.join(d, "group_stats.jsonl")
        if os.path.isfile(sp):
            with open(sp) as f:
                stats = [json.loads(line) for line in f if line.strip()]
        ev = None
        ep = os.path.join(d, "eval.json")
        if os.path.isfile(ep):
            with open(ep) as f:
                ev = json.load(f)
        runs.append({"name": name, "config": cfg, "stats": stats, "eval": ev})
    return runs


def smooth(xs, k=20):
    """Trailing moving average. Reward-call logs are noisy; raw curves are unreadable."""
    out, acc = [], 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= k:
            acc -= xs[i - k]
        out.append(acc / min(i + 1, k))
    return out


def fig_wasted_compute(runs, out_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for run in runs:
        if not run["stats"]:
            continue
        ys = smooth([r["frac_rollouts_wasted"] for r in run["stats"]])
        ax.plot(range(len(ys)), ys,
                label=f"G={run['config']['num_generations']} ({run['name']})")
        plotted = True
    if not plotted:
        return False
    ax.set_xlabel("reward calls (training progress)")
    ax.set_ylabel("fraction of rollouts with zero gradient")
    ax.set_ylim(0, 1)
    ax.set_title("Wasted compute: rollouts in zero-variance groups")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_wasted_compute.png"), dpi=200)
    plt.close(fig)
    return True


def fig_group_size(runs, out_dir, grader="strict", bench="gsm8k"):
    import matplotlib.pyplot as plt

    # G -> list of final accuracies (one per seed)
    by_g = defaultdict(list)
    for run in runs:
        ev = run["eval"]
        if not ev or bench not in ev.get("benchmarks", {}):
            continue
        acc = ev["benchmarks"][bench]["accuracy_by_grader"].get(grader)
        if acc is not None:
            by_g[run["config"]["num_generations"]].append(acc)
    if not by_g:
        return False

    gs = sorted(by_g)
    means = [sum(by_g[g]) / len(by_g[g]) for g in gs]
    los = [min(by_g[g]) for g in gs]
    his = [max(by_g[g]) for g in gs]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(gs, means, "o-", label=f"mean over {max(len(v) for v in by_g.values())} seed(s)")
    ax.fill_between(gs, los, his, alpha=0.2, label="seed min-max")
    ax.set_xscale("log", base=2)
    ax.set_xticks(gs, [str(g) for g in gs])
    ax.set_xlabel("group size G (matched total rollout budget)")
    ax.set_ylabel(f"{bench} accuracy ({grader} grader)")
    ax.set_title("Accuracy vs group size at fixed compute")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_group_size.png"), dpi=200)
    plt.close(fig)
    return True


def grader_table(runs, out_dir, bench="gsm8k"):
    rows = []
    graders = None
    for run in runs:
        ev = run["eval"]
        if not ev or bench not in ev.get("benchmarks", {}):
            continue
        accs = ev["benchmarks"][bench]["accuracy_by_grader"]
        graders = graders or list(accs)
        rows.append((run["name"], accs))
    if not rows or graders is None:
        return False
    lines = [f"| run | {' | '.join(graders)} |",
             f"|-----|{'------|' * len(graders)}"]
    for name, accs in rows:
        lines.append(f"| {name} | " + " | ".join(f"{accs.get(g, 0):.4f}" for g in graders) + " |")
    # Flag ranking flips: the point of the table.
    rank = {g: [n for n, _ in sorted(rows, key=lambda r: -r[1].get(g, 0))] for g in graders}
    flips = len({tuple(v) for v in rank.values()}) > 1
    lines.append("")
    lines.append(f"**Ranking {'FLIPS' if flips else 'is stable'} across graders.**")
    with open(os.path.join(out_dir, "fig3_grader_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return True


def selftest():
    import tempfile

    d = tempfile.mkdtemp()
    runs_dir = os.path.join(d, "runs")
    out_dir = os.path.join(d, "figures")
    os.makedirs(out_dir)
    # Two fake runs: G=4 (2 seeds worth collapsed to 1) and G=16, with a grader flip.
    for name, g, wasted, strict_acc, lenient_acc in [
        ("g4_seed0", 4, 0.7, 0.30, 0.31),
        ("g16_seed0", 16, 0.5, 0.28, 0.35),   # lenient grader flips the winner
    ]:
        rd = os.path.join(runs_dir, name)
        os.makedirs(rd)
        with open(os.path.join(rd, "config.json"), "w") as f:
            json.dump({"num_generations": g, "seed": 0}, f)
        with open(os.path.join(rd, "group_stats.jsonl"), "w") as f:
            for i in range(50):
                f.write(json.dumps({"frac_rollouts_wasted": wasted - i * 0.002}) + "\n")
        with open(os.path.join(rd, "eval.json"), "w") as f:
            json.dump({"benchmarks": {"gsm8k": {"n": 10, "accuracy_by_grader": {
                "strict": strict_acc, "last_number": lenient_acc, "boxed": strict_acc,
            }}}}, f)

    runs = load_runs(runs_dir)
    assert len(runs) == 2
    assert fig_wasted_compute(runs, out_dir)
    assert fig_group_size(runs, out_dir)
    assert grader_table(runs, out_dir)
    table = open(os.path.join(out_dir, "fig3_grader_table.md")).read()
    assert "FLIPS" in table, table   # the fake data contains a deliberate flip
    assert smooth([1.0] * 5) == [1.0] * 5
    for fn in ("fig1_wasted_compute.png", "fig2_group_size.png"):
        assert os.path.getsize(os.path.join(out_dir, fn)) > 0
    print("analyze selftest: all checks passed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--out_dir", default="figures")
    args = p.parse_args()

    if args.selftest:
        selftest()
        return

    runs = load_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"no runs with config.json found under {args.runs_dir}/")
    os.makedirs(args.out_dir, exist_ok=True)
    made = {
        "fig1 (wasted compute)": fig_wasted_compute(runs, args.out_dir),
        "fig2 (group size)": fig_group_size(runs, args.out_dir),
        "fig3 (grader table)": grader_table(runs, args.out_dir),
    }
    for k, v in made.items():
        print(f"{k}: {'written' if v else 'skipped (no data yet)'}")


if __name__ == "__main__":
    main()

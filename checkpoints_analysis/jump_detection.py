"""Behavioural jump detection for the *_main runs (metrics.csv only).

    python -m checkpoints_analysis.jump_detection            # writes logs/run_results.md
    python -m checkpoints_analysis.jump_detection --out -    # print to stdout

Reads only config.json, metrics.csv and the checkpoints/ directory listing of
each runs/*_main directory. It never opens signals.csv and never loads a
checkpoint.

Jump definitions (per run, on the task's behavioural metric):
  t_thr  = first evaluated step s such that metric(t) >= thr for every
           evaluated step t >= s (i.e. it crosses thr and never drops back).
           None if the metric is below thr at the last evaluated step.
  t50, t90 use thr = 0.5 and 0.9; width = t90 - t10 with t10 defined the same way.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_OUT = REPO_ROOT / "logs" / "run_results.md"

TASKS = {
    # task name -> (behavioural metric column, expected checkpoint file count)
    "induction": ("heldout_acc", 201),
    "icl": ("fresh_map_acc", 301),
    "grokking": ("val_acc", 151),
}
TASK_ORDER = ["induction", "icl", "grokking"]
THRESHOLDS = (0.1, 0.5, 0.9)
# Tasks whose metric starts above 0.1 (ICL: chance = 1/n_labels = 0.125), so t10 and
# width (t90 - t10) do not describe a transition. Reported as "—".
NO_T10_TASKS = {"icl"}

# Context supplied by the experimenter that cannot be read from the files.
# Keyed by (task, seed). Shown next to the file-based exclusion reasons.
EXPERIMENTER_NOTES = {
    ("grokking", 5): "experimenter reports training stopped near step 4,500 when the disk filled",
}


def read_metrics(path: Path, column: str):
    """Return (last step in file, [(step, value)] for rows where `column` is a number)."""
    last_step, series = None, []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            step = int(float(row["step"]))
            last_step = step if last_step is None else max(last_step, step)
            raw = row.get(column, "")
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isnan(val):
                series.append((step, val))
    series.sort()
    return last_step, series


def stays_above_step(series, thr):
    """First step after which the metric stays >= thr for the rest of the run."""
    first = None
    for step, val in reversed(series):
        if val >= thr:
            first = step
        else:
            break
    return first


def first_crossing_step(series, thr):
    """First evaluated step at which the metric is >= thr (sensitivity check)."""
    return next((step for step, val in series if val >= thr), None)


def inspect_run(run_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    task = config["task"]["name"]
    metric, expected_ckpts = TASKS[task]
    train_steps = int(config["training"]["train_steps"])

    ckpt_dir = run_dir / "checkpoints"
    files = [p for p in ckpt_dir.iterdir() if p.is_file()] if ckpt_dir.is_dir() else []
    empty = sorted(p.name for p in files if p.stat().st_size == 0)

    info = {
        "run": run_dir.name, "task": task, "seed": config.get("seed"), "config": config,
        "metric": metric, "train_steps": train_steps,
        "n_ckpts": len(files), "expected_ckpts": expected_ckpts, "empty_ckpts": empty,
        "last_ckpt": max((p.name for p in files), default=None),
        "final_step": None, "series": [], "reasons": [],
    }

    if run_dir.name.endswith("_INCOMPLETE"):
        info["reasons"].append("run directory marked _INCOMPLETE")

    metrics_path = run_dir / "metrics.csv"
    if metrics_path.is_file():
        info["final_step"], info["series"] = read_metrics(metrics_path, metric)
        if info["final_step"] is None or info["final_step"] < train_steps:
            info["reasons"].append(
                f"metrics.csv ends at step {info['final_step']}, configured train_steps={train_steps}")
        if not info["series"]:
            info["reasons"].append(f"metrics.csv has no values for {metric}")
    else:
        info["reasons"].append(f"no metrics.csv (configured train_steps={train_steps})")

    if len(files) < expected_ckpts:
        info["reasons"].append(f"{len(files)} checkpoint files, expected {expected_ckpts} "
                               f"(last file: {info['last_ckpt']})")
    if empty:
        info["reasons"].append(f"zero-byte checkpoint file(s): {', '.join(empty)}")

    if not info["reasons"]:
        s = info["series"]
        for prefix, fn in (("t", stays_above_step), ("fc", first_crossing_step)):
            for thr in THRESHOLDS:
                info[f"{prefix}{int(thr * 100)}"] = fn(s, thr)
            if task in NO_T10_TASKS:
                info[f"{prefix}10"] = None
            info[f"{prefix}width"] = diff(info[f"{prefix}90"], info[f"{prefix}10"])
            info[f"{prefix}rise"] = diff(info[f"{prefix}90"], info[f"{prefix}50"])
        info["final_value"] = s[-1][1]
    return info


def diff(a, b):
    return a - b if a is not None and b is not None else None


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.4f}" if not x.is_integer() else f"{int(x)}"
    return str(x)


def agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return "n/a", "n/a", "n/a", 0
    return f"{mean(vals):.1f}", fmt(min(vals)), fmt(max(vals)), len(vals)


def build_report(runs: list[dict]) -> str:
    L = []
    L.append("# Main-run behavioural results\n")
    L.append("Generated by `python -m checkpoints_analysis.jump_detection` from "
             "`runs/*_main/{config.json,metrics.csv,checkpoints/}` only. "
             "No signals.csv was opened and no checkpoint was loaded.\n")
    L.append("Jump definition: for threshold θ, t_θ is the first evaluated step after which the "
             "metric stays ≥ θ at every later evaluated step until the end of the run "
             "(primary). As a sensitivity check, FC t_θ is the first evaluated step at which "
             "the metric is ≥ θ, ignoring later dips. "
             "t50 and t90 use θ = 0.5 and 0.9; width = t90 − t10; rise = t90 − t50. "
             "t10 and width are not reported for ICL, whose accuracy starts above 0.1. "
             "Steps are training steps; "
             "resolution is the eval interval (listed under settings).\n")
    L.append("Metrics: induction `heldout_acc`, ICL `fresh_map_acc`, grokking `val_acc`.\n")

    # --- inventory
    L.append("## Run inventory\n")
    L.append("| Run | Task | Seed | Final step in metrics.csv | train_steps | Checkpoint files | Expected | Status |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in runs:
        status = "EXCLUDED" if r["reasons"] else "ok"
        L.append(f"| {r['run']} | {r['task']} | {r['seed']} | {fmt(r['final_step'])} | "
                 f"{r['train_steps']} | {r['n_ckpts']} | {r['expected_ckpts']} | {status} |")
    L.append("")

    # --- exclusions
    L.append("## Exclusions\n")
    excluded = [r for r in runs if r["reasons"]]
    if not excluded:
        L.append("None.\n")
    for r in excluded:
        reasons = list(r["reasons"])
        note = EXPERIMENTER_NOTES.get((r["task"], r["seed"]))
        if note:
            reasons.append(note)
        L.append(f"- `{r['run']}` ({r['task']}, seed {r['seed']}): " + "; ".join(reasons) + ".")
    L.append("")

    # --- per-run jumps
    kept = [r for r in runs if not r["reasons"]]

    def cell(r, key):
        if r["task"] in NO_T10_TASKS and key.endswith(("10", "width")):
            return "—"
        return fmt(r[key])

    L.append("## Per-run jump steps\n")
    L.append("Primary = stays-above; FC = first crossing (sensitivity check). "
             "Rise = t90 − t50; width = t90 − t10. "
             "“—” = not reported for this task (see Caveats).\n")
    cols = ["t10", "t50", "t90", "width", "rise", "fct10", "fct50", "fct90", "fcwidth", "fcrise"]
    keys = ["t10", "t50", "t90", "twidth", "trise", "fc10", "fc50", "fc90", "fcwidth", "fcrise"]
    L.append("| Task | Seed | t10 | t50 | t90 | width | rise | FC t10 | FC t50 | FC t90 | "
             "FC width | FC rise | final metric |")
    L.append("|---|---|" + "---|" * (len(cols) + 1))
    for task in TASK_ORDER:
        for r in sorted((r for r in kept if r["task"] == task), key=lambda r: r["seed"]):
            L.append(f"| {task} | {r['seed']} | " + " | ".join(cell(r, k) for k in keys)
                     + f" | {r['final_value']:.4f} |")
    L.append("")

    # --- per-task summary
    def summary_table(title, prefix):
        L.append(f"### {title}\n")
        names = [("50", "t50"), ("90", "t90"), ("width", "width"), ("rise", "rise")]
        L.append("| Task | n seeds | " + " | ".join(
            f"{lab} {s}" for _, lab in names for s in ("mean", "min", "max")) + " |")
        L.append("|---|---|" + "---|" * (3 * len(names)))
        for task in TASK_ORDER:
            rs = [r for r in kept if r["task"] == task]
            if not rs:
                continue
            cells = []
            for suffix, _ in names:
                key = f"{prefix}{suffix}"
                if task in NO_T10_TASKS and suffix == "width":
                    cells += ["—"] * 3
                    continue
                m, lo, hi, n = agg(r[key] for r in rs)
                if n != len(rs):
                    m += f" (n={n})"
                cells += [m, lo, hi]
            seeds = ",".join(str(r["seed"]) for r in sorted(rs, key=lambda r: r["seed"]))
            L.append(f"| {task} | {len(rs)} (seeds {seeds}) | " + " | ".join(cells) + " |")
        L.append("")

    L.append("## Per-task summary across seeds (mean / min / max)\n")
    summary_table("Primary: stays-above", "t")
    summary_table("Sensitivity check: first crossing", "fc")

    # --- caveats (computed, not hand-written)
    L.append("## Caveats\n")
    caveats = []
    for r in kept:
        s = r["series"]
        first_eval = next((st for st, _ in s if st > s[0][0]), None)
        raw_t10 = stays_above_step(s, 0.1)
        if raw_t10 is not None and raw_t10 <= (first_eval or 0):
            n_lab = r["config"]["task"].get("n_labels")
            chance = f" (chance = 1/{n_lab} = {1 / n_lab:.3f})" if n_lab else ""
            action = ("t10 and width are not reported for this task"
                      if r["task"] in NO_T10_TASKS else
                      "its t10 and width measure time from the start of training, not a transition")
            caveats.append(
                f"{r['task']} seed {r['seed']}: `{r['metric']}` is already ≥ 0.1 at the first "
                f"evaluation after step 0 (step {first_eval}){chance}; {action}.")
        for thr in (0.5, 0.9):
            first_cross = next((st for st, v in s if v >= thr), None)
            t_stay = r[f"t{int(thr * 100)}"]
            if first_cross is not None and t_stay != first_cross:
                dips = [(st, v) for st, v in s if first_cross < st and v < thr]
                dip_txt = ", ".join(f"step {st}: {v:.3f}" for st, v in dips[:5])
                caveats.append(
                    f"{r['task']} seed {r['seed']}: first reaches {thr:g} at step {first_cross} but "
                    f"drops below it again ({dip_txt}{', …' if len(dips) > 5 else ''}), so "
                    f"t{int(thr * 100)} = {t_stay} under the stays-above definition.")
    L += [f"- {c}" for c in caveats] if caveats else ["None."]
    L.append("")

    # --- settings
    L.append("## Settings per task (from config.json)\n")
    L.append("All runs of a task were checked for identical config.json apart from `seed`; "
             "any key that differs is listed per seed below its table.\n")
    for task in TASK_ORDER:
        rs = sorted((r for r in runs if r["task"] == task), key=lambda r: r["seed"])
        if not rs:
            continue
        flats = [flatten(r["config"]) for r in rs]
        keys = sorted(set().union(*flats) - {"seed"})
        same = [k for k in keys if len({json.dumps(f.get(k)) for f in flats}) == 1]
        diff = [k for k in keys if k not in same]
        L.append(f"### {task} (metric `{TASKS[task][0]}`, seeds "
                 f"{', '.join(str(r['seed']) for r in rs)})\n")
        L.append("| Key | Value |")
        L.append("|---|---|")
        for k in same:
            L.append(f"| `{k}` | `{json.dumps(flats[0].get(k))}` |")
        L.append("")
        for k in diff:
            vals = ", ".join(f"seed {r['seed']}: `{json.dumps(f.get(k))}`" for r, f in zip(rs, flats))
            L.append(f"- `{k}` differs: {vals}")
        if diff:
            L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output path, or '-' for stdout")
    args = ap.parse_args()

    # *_main_INCOMPLETE dirs are main runs set aside by hand; list them so the
    # exclusion stays documented, and always exclude them.
    run_dirs = sorted(p for pat in ("*_main", "*_main_INCOMPLETE")
                      for p in args.runs_dir.glob(pat) if p.is_dir())
    runs = [inspect_run(p) for p in run_dirs]
    runs.sort(key=lambda r: (TASK_ORDER.index(r["task"]), r["seed"]))
    report = build_report(runs)
    if args.out == "-":
        print(report)
    else:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"wrote {args.out} ({len(runs)} runs, "
              f"{sum(not r['reasons'] for r in runs)} kept)")


if __name__ == "__main__":
    main()

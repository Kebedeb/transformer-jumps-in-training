"""Pre-registered precursor analysis (docs/hypothesis.md, commit-locked).

    python -m checkpoints_analysis.precursor_analysis            # pre-registered defaults

Inputs: runs/*-{induction,icl,grokking}_main/{config.json,metrics.csv,signals.csv}
(directories ending _INCOMPLETE are not matched). Outputs: the report
(--out, default logs/precursor_results.md) and one PNG per task (--fig-dir,
default logs/figures/). Every analysis parameter is a CLI argument whose default
is the pre-registered value, so the S1 sensitivity grid can be run later by
changing arguments only.

Dependencies: numpy, matplotlib (no pandas/scipy).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from checkpoints_analysis.jump_detection import (
    TASKS as JD_TASKS, first_crossing_step, read_metrics, stays_above_step,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_ORDER = ["induction", "icl", "grokking"]
TASK_LABEL = {"induction": "Induction", "icl": "ICL", "grokking": "Grokking"}
SIGNALS = [
    "weight_norm_total",
    "resid{L}_eff_rank",
    "resid{L}_part_ratio",
    "1 - resid{L}_cka_prev",
    "mean attn_L*H*_entropy",
    "max attn_L*H*_prev_token",
]
POS_CONTROL = "attn_max_pattern"
POS_CONTROL_TASKS = ["induction", "icl"]
# Categorical slots 1-5 of the dataviz reference palette, fixed per seed.
SEED_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100", 5: "#e87ba4"}


# ----------------------------------------------------------------------------- args
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Pre-registered precursor analysis.")
    io = ap.add_argument_group("paths")
    io.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    io.add_argument("--run-results", type=Path, default=REPO_ROOT / "logs" / "run_results.md",
                    help="jump table to check t50 against (step 1)")
    io.add_argument("--out", type=Path, default=REPO_ROOT / "logs" / "precursor_results.md")
    io.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "logs" / "figures")
    p = ap.add_argument_group("pre-registered parameters (defaults = pre-registered values)")
    p.add_argument("--t50-threshold", type=float, default=0.5)
    p.add_argument("--t50-mode", choices=["stays_above", "first_crossing"], default="stays_above")
    p.add_argument("--smooth-window", type=int, default=5,
                   help="centered rolling median window in checkpoints (min_periods=1)")
    p.add_argument("--ref-lo", type=float, default=0.05, help="reference window start, fraction of t50")
    p.add_argument("--ref-hi", type=float, default=0.30, help="reference window end, fraction of t50")
    p.add_argument("--mad-scale", type=float, default=1.4826)
    p.add_argument("--sigma-floor-rel", type=float, default=1e-6)
    p.add_argument("--sigma-floor-abs", type=float, default=1e-12)
    p.add_argument("--band-k", type=float, default=3.0, help="band half-width in sigma")
    p.add_argument("--min-lead-intervals", type=float, default=2.0,
                   help="minimum lead in checkpoint intervals")
    p.add_argument("--min-lead-frac", type=float, default=0.05, help="minimum lead as fraction of t50")
    p.add_argument("--seeds-allowed-to-fail", type=int, default=1,
                   help="precursor needs >= n_seeds - this many qualifying seeds")
    p.add_argument("--max-lead-ratio", type=float, default=3.0,
                   help="shared candidate: max/min of median normalized leads <= this")
    return ap.parse_args(argv)


PARAM_KEYS = ["t50_threshold", "t50_mode", "smooth_window", "ref_lo", "ref_hi", "mad_scale",
              "sigma_floor_rel", "sigma_floor_abs", "band_k", "min_lead_intervals",
              "min_lead_frac", "seeds_allowed_to_fail", "max_lead_ratio"]


# ----------------------------------------------------------------------------- io
def read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames
    out = {}
    for c in cols:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[c]))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        out[c] = np.array(vals, dtype=float)
    return out


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(git failed: {e})"


def parse_run_results(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    """Per-run jump table of logs/run_results.md -> {(task, seed): {column: cell}}."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index("## Per-run jump steps")
    header, table = None, {}
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
        elif set(line) <= set("|- "):
            continue
        else:
            row = dict(zip(header, cells))
            table[(row["Task"], int(row["Seed"]))] = row
    return table


# ----------------------------------------------------------------------------- signals
def build_signals(cols: dict[str, np.ndarray], L: int):
    """Return {signal name: (steps, values)} and notes on how each was built."""
    steps = cols["step"]
    ent = sorted(c for c in cols if re.fullmatch(r"attn_L\d+H\d+_entropy", c))
    prev = sorted(c for c in cols if re.fullmatch(r"attn_L\d+H\d+_prev_token", c))
    raw = {
        "weight_norm_total": cols["weight_norm_total"],
        "resid{L}_eff_rank": cols[f"resid{L}_eff_rank"],
        "resid{L}_part_ratio": cols[f"resid{L}_part_ratio"],
        "1 - resid{L}_cka_prev": 1.0 - cols[f"resid{L}_cka_prev"],
        "mean attn_L*H*_entropy": np.mean(np.vstack([cols[c] for c in ent]), axis=0),
        "max attn_L*H*_prev_token": np.max(np.vstack([cols[c] for c in prev]), axis=0),
    }
    if POS_CONTROL in cols:
        raw[POS_CONTROL] = cols[POS_CONTROL]
    out, nan_notes = {}, []
    for name, v in raw.items():
        bad = np.isnan(v)
        if name == "1 - resid{L}_cka_prev":
            # pre-registered: drop the step-0 NaN
            drop = bad & (steps == steps.min())
            extra = bad & ~drop
        else:
            drop = np.zeros_like(bad)
            extra = bad
        if extra.any():
            nan_notes.append(f"{name}: NaN at steps {steps[extra].astype(int).tolist()} (kept as NaN)")
        out[name] = (steps[~drop], v[~drop])
    return out, {"entropy_cols": ent, "prev_token_cols": prev}, nan_notes


def rolling_median_centered(v: np.ndarray, window: int) -> np.ndarray:
    """pandas rolling(window, center=True, min_periods=1).median() equivalent (NaN-skipping)."""
    n, half_lo = len(v), (window - 1) // 2
    half_hi = window - 1 - half_lo
    # pandas centers with the extra element on the right? For odd windows it is symmetric;
    # for even windows pandas puts the extra element on the left of the window label.
    if window % 2 == 0:
        half_lo, half_hi = window // 2, window // 2 - 1
    out = np.full(n, np.nan)
    for i in range(n):
        w = v[max(0, i - half_lo): min(n, i + half_hi + 1)]
        w = w[~np.isnan(w)]
        if len(w):
            out[i] = np.median(w)
    return out


def detect_onset(steps, smoothed, t50, a):
    """Onset rule from the pre-registration (see module docstring / report section 8)."""
    res = {"m": None, "sigma": None, "n_ref": 0, "sigma_floored": False, "onset": None,
           "direction": 0, "inside_at_t50": None, "t50_in_series": bool(np.any(steps == t50)),
           "onset_at_first_candidate": False}
    ref = (steps >= a.ref_lo * t50) & (steps <= a.ref_hi * t50) & ~np.isnan(smoothed)
    res["n_ref"] = int(ref.sum())
    if res["n_ref"] == 0:
        return res
    m = float(np.median(smoothed[ref]))
    mad = float(np.median(np.abs(smoothed[ref] - m)))
    floor = a.sigma_floor_rel * abs(m) + a.sigma_floor_abs
    sigma = max(a.mad_scale * mad, floor)
    res.update(m=m, sigma=sigma, sigma_floored=a.mad_scale * mad < floor)
    lo, hi = m - a.band_k * sigma, m + a.band_k * sigma

    at_t50 = np.where(steps == t50)[0]
    if len(at_t50) == 0:
        return res
    v50 = smoothed[at_t50[0]]
    res["direction"] = int(np.sign(v50 - m)) if not np.isnan(v50) else 0
    cand = np.where((steps > a.ref_hi * t50) & (steps <= t50))[0]
    outside = (smoothed < lo) | (smoothed > hi)  # NaN -> False (treated as not outside)
    res["inside_at_t50"] = not bool(outside[at_t50[0]])
    if res["inside_at_t50"] or len(cand) == 0:
        return res
    # walk back from t50 while the smoothed signal stays outside the band
    onset_idx = None
    for idx in cand[::-1]:
        if outside[idx]:
            onset_idx = idx
        else:
            break
    if onset_idx is not None:
        res["onset"] = int(steps[onset_idx])
        res["onset_at_first_candidate"] = onset_idx == cand[0]
    return res


# ----------------------------------------------------------------------------- stats
def rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


# ----------------------------------------------------------------------------- main
def load_runs(a):
    runs, inconsistencies = [], []
    for task in TASK_ORDER:
        for d in sorted(a.runs_dir.glob(f"*-{task}_main")):
            if not d.is_dir() or d.name.endswith("_INCOMPLETE"):
                continue
            cfg = json.loads((d / "config.json").read_text())
            metric = JD_TASKS[task][0]
            _, series = read_metrics(d / "metrics.csv", metric)
            fn = stays_above_step if a.t50_mode == "stays_above" else first_crossing_step
            t50 = fn(series, a.t50_threshold)
            cols = read_csv_columns(d / "signals.csv")
            runs.append({"task": task, "seed": int(cfg["seed"]), "dir": d.name, "config": cfg,
                         "L": int(cfg["model"]["n_layers"]), "t50": t50, "cols": cols})
    return runs, inconsistencies


def check_t50(runs, a):
    table = parse_run_results(a.run_results)
    col = "t50" if a.t50_mode == "stays_above" else "FC t50"
    mismatches = []
    for r in runs:
        row = table.get((r["task"], r["seed"]))
        ref = row.get(col) if row else None
        ok = ref is not None and r["t50"] is not None and ref == str(r["t50"])
        if not ok:
            mismatches.append(f"{r['task']} seed {r['seed']} ({r['dir']}): computed t50={r['t50']}, "
                              f"{a.run_results.name} {col}={ref}")
    run_keys = {(r["task"], r["seed"]) for r in runs}
    for k in table:
        if k not in run_keys:
            mismatches.append(f"{k[0]} seed {k[1]} is in {a.run_results.name} but not among the runs")
    if a.t50_threshold != 0.5:
        mismatches.append("t50 check skipped: --t50-threshold is not 0.5, run_results.md has no "
                          "matching column")
        return [], mismatches  # informational only
    return mismatches, []


def analyse(runs, a):
    rows, notes = [], {"nan": [], "spacing": [], "t50_not_ckpt": [], "floored": [],
                       "first_candidate": [], "few_ref": [], "cols": {}}
    for r in runs:
        steps_all = r["cols"]["step"]
        diffs = np.unique(np.diff(steps_all))
        interval = float(np.median(np.diff(steps_all)))
        if len(diffs) != 1:
            notes["spacing"].append(f"{r['task']} seed {r['seed']}: step spacings {diffs.tolist()}")
        sigs, colinfo, nan_notes = build_signals(r["cols"], r["L"])
        notes["cols"][r["task"]] = colinfo
        notes["nan"] += [f"{r['task']} seed {r['seed']}: {n}" for n in nan_notes]
        r["interval"] = interval
        r["smoothed"] = {}
        names = SIGNALS + ([POS_CONTROL] if r["task"] in POS_CONTROL_TASKS else [])
        for name in names:
            steps, v = sigs[name]
            sm = rolling_median_centered(v, a.smooth_window)
            res = detect_onset(steps, sm, r["t50"], a)
            r["smoothed"][name] = (steps, sm, res)
            min_lead = max(a.min_lead_intervals * interval, a.min_lead_frac * r["t50"])
            lead = r["t50"] - res["onset"] if res["onset"] is not None else None
            qual = lead is not None and lead >= min_lead
            rows.append({"task": r["task"], "signal": name, "seed": r["seed"], "t50": r["t50"],
                         "onset": res["onset"], "lead": lead,
                         "norm_lead": lead / r["t50"] if lead is not None else None,
                         "direction": res["direction"], "min_lead": min_lead, "qualifies": qual,
                         **{k: res[k] for k in ("m", "sigma", "n_ref", "sigma_floored",
                                                "inside_at_t50", "onset_at_first_candidate")}})
            tag = f"{r['task']} seed {r['seed']} `{sig_label(name)}`"
            if not res["t50_in_series"]:
                notes["t50_not_ckpt"].append(tag)
            if res["sigma_floored"]:
                notes["floored"].append(tag)
            if res["onset_at_first_candidate"]:
                notes["first_candidate"].append(tag)
            if res["n_ref"] < 3:
                notes["few_ref"].append(f"{tag}: {res['n_ref']} checkpoints")
    return rows, notes


def verdicts(rows, runs, a):
    out = {}
    for task in TASK_ORDER:
        n = sum(r["task"] == task for r in runs)
        for sig in SIGNALS:
            rs = [x for x in rows if x["task"] == task and x["signal"] == sig]
            q = [x for x in rs if x["qualifies"]]
            dirs = sorted({x["direction"] for x in q})
            need = n - a.seeds_allowed_to_fail
            is_prec = len(q) >= need and len(dirs) == 1
            out[(task, sig)] = {
                "n_seeds": n, "n_qual": len(q), "need": need, "precursor": is_prec,
                "direction": (dirs[0] if len(dirs) == 1 else ("mixed" if dirs else None)),
                "median_norm_lead": float(np.median([x["norm_lead"] for x in q])) if q else None,
                "one_seed_short": len(q) == need - 1,
            }
    return out


def cross_task(ver, a):
    out = []
    for sig in SIGNALS:
        vs = [ver[(t, sig)] for t in TASK_ORDER]
        in_all = all(v["precursor"] for v in vs)
        dirs = {v["direction"] for v in vs}
        same_dir = in_all and len(dirs) == 1
        leads = [v["median_norm_lead"] for v in vs]
        ratio = (max(leads) / min(leads)) if in_all and min(leads) > 0 else None
        cand = in_all and same_dir and ratio is not None and ratio <= a.max_lead_ratio
        out.append({"signal": sig, "n_tasks": sum(v["precursor"] for v in vs),
                    "tasks": [t for t, v in zip(TASK_ORDER, vs) if v["precursor"]],
                    "in_all": in_all, "same_dir": same_dir, "ratio": ratio, "candidate": cand})
    return out


# ----------------------------------------------------------------------------- figures
def make_figures(runs, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a.fig_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for task in TASK_ORDER:
        trs = sorted((r for r in runs if r["task"] == task), key=lambda r: r["seed"])
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
        for ax, sig in zip(axes.flat, SIGNALS):
            for r in trs:
                steps, sm, res = r["smoothed"][sig]
                if res["m"] is None:
                    continue
                x = (steps - r["t50"]) / r["t50"]
                y = sm - res["m"]
                c = SEED_COLORS.get(r["seed"], "#6250d6")
                ax.plot(x, y, color=c, lw=1.5, label=f"seed {r['seed']}")
                if res["onset"] is not None:
                    i = int(np.where(steps == res["onset"])[0][0])
                    ax.plot(x[i], y[i], "o", ms=8, color=c, mec="white", mew=1.5, zorder=5)
            ax.axvline(0, color="#555555", lw=1, ls="--")
            ax.set_title(sig.replace("{L}", str(trs[0]["L"])), fontsize=11)
            ax.set_xlabel("(step − t50) / t50")
            ax.set_ylabel("smoothed − reference median m")
            ax.grid(alpha=0.25, lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        handles.append(plt.Line2D([], [], marker="o", ls="", color="#777777", mec="white", ms=8))
        labels.append("onset")
        fig.legend(handles, labels, loc="outside upper right", ncol=len(labels), frameon=False)
        fig.suptitle(f"{TASK_LABEL[task]}: candidate signals around t50 "
                     f"(dashed line = t50; dot = onset)", fontsize=13, x=0.01, ha="left")
        p = a.fig_dir / f"precursor_{task}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths[task] = p
    return paths


# ----------------------------------------------------------------------------- report
def f(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        return f"{x:.{nd}f}" if not x.is_integer() else str(int(x))
    return str(x)


def dirs(d):
    return {1: "+", -1: "−", 0: "0", None: "—", "mixed": "mixed"}.get(d, str(d))


def runs_L(runs, task):
    Ls = sorted({r["L"] for r in runs if r["task"] == task})
    return Ls[0] if len(Ls) == 1 else "/".join(map(str, Ls))


def sig_label(sig):
    return sig.replace("{L}", "L")


def write_report(runs, rows, notes, ver, cross, figs, a, t50_info):
    L = []
    rel = lambda p: Path(p).resolve().relative_to(a.out.parent.resolve()).as_posix()  # noqa: E731
    n_runs = {t: sum(r["task"] == t for r in runs) for t in TASK_ORDER}
    prec = {t: [s for s in SIGNALS if ver[(t, s)]["precursor"]] for t in TASK_ORDER}
    cands = [c for c in cross if c["candidate"]]
    pc = [x for x in rows if x["signal"] == POS_CONTROL]

    L.append("# Precursor analysis results (pre-registered, defaults)\n")
    L.append("Generated by `python -m checkpoints_analysis.precursor_analysis`. "
             "Do not edit by hand; re-run the script.\n")

    # 0 summary
    L.append("## 0. Summary\n")
    L.append(f"- Runs analysed: induction {n_runs['induction']}, ICL {n_runs['icl']}, "
             f"grokking {n_runs['grokking']}; every t50 matched logs/run_results.md.")
    for t in TASK_ORDER:
        L.append(f"- {TASK_LABEL[t]}: precursor in {len(prec[t])} of 6 signals"
                 + (f" ({', '.join(f'`{sig_label(s)}`' for s in prec[t])})." if prec[t] else "."))
    if cands:
        L.append("- Shared candidate (specificity pending): "
                 + ", ".join(f"`{sig_label(c['signal'])}`" for c in cands) + ".")
    else:
        L.append("- No signal meets the shared-candidate rule (precursor in all three tasks, same "
                 f"direction, lead ratio ≤ {f(a.max_lead_ratio)}).")
    pc_q = sum(x["qualifies"] for x in pc)
    pc_none = sum(x["onset"] is None for x in pc)
    L.append(f"- Positive control `{POS_CONTROL}`: {pc_q} of {len(pc)} induction/ICL runs qualify; "
             f"{pc_none} have no onset.")
    L.append("- Controls have not been run; specificity is not assessed.\n")

    # 1 provenance
    L.append("## 1. Provenance\n")
    L.append(f"- HEAD commit: `{git('rev-parse', 'HEAD')}`")
    L.append(f"- Last commit touching docs/hypothesis.md: "
             f"`{git('log', '-1', '--format=%H %ad %s', '--', 'docs/hypothesis.md')}`")
    dirty = git("status", "--porcelain", "--", "docs/hypothesis.md", "checkpoints_analysis")
    L.append(f"- Uncommitted changes in docs/hypothesis.md or checkpoints_analysis/: "
             f"{'none' if not dirty else '`' + dirty.replace(chr(10), '; ') + '`'}")
    L.append(f"- Python {sys.version.split()[0]}, numpy {np.__version__}")
    L.append("- Runs: " + ", ".join(f"`{r['dir']}`" for r in runs))
    L.append("- Parameters:\n")
    L.append("| Parameter | Value |")
    L.append("|---|---|")
    for k in PARAM_KEYS:
        L.append(f"| `--{k.replace('_', '-')}` | `{getattr(a, k)}` |")
    L.append("")

    # 2 per-run
    L.append("## 2. Per-run table\n")
    L.append("Lead = t50 − onset; normalized lead = lead / t50; qualifies = onset exists and "
             "lead ≥ max(2 × checkpoint interval, 0.05 × t50). Direction = sign(smoothed signal at "
             "t50 − m).\n")
    L.append("| Task | Signal | Seed | t50 | Onset | Lead | Norm. lead | Min lead | Direction | Qualifies |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for t in TASK_ORDER:
        for s in SIGNALS:
            for x in sorted((x for x in rows if x["task"] == t and x["signal"] == s),
                            key=lambda x: x["seed"]):
                L.append(f"| {t} | `{sig_label(s)}` | {x['seed']} | {x['t50']} | {f(x['onset'])} | "
                         f"{f(x['lead'])} | {f(x['norm_lead'])} | {f(x['min_lead'])} | "
                         f"{dirs(x['direction'])} | {f(x['qualifies'])} |")
    L.append("")

    # 3 per-task verdict
    L.append("## 3. Per-task verdict\n")
    L.append("Precursor = at least n − 1 seeds qualify and all qualifying seeds share one direction. "
             "“One seed short” = exactly one fewer qualifying seed than required (the "
             "pre-registered “fails by one seed” condition).\n")
    L.append("| Signal | Task | Precursor | Qualifying / seeds | Direction (qualifying) | "
             "Median norm. lead | One seed short |")
    L.append("|---|---|---|---|---|---|---|")
    for s in SIGNALS:
        for t in TASK_ORDER:
            v = ver[(t, s)]
            L.append(f"| `{sig_label(s)}` | {t} | {f(v['precursor'])} | {v['n_qual']} / {v['n_seeds']} "
                     f"(need {v['need']}) | {dirs(v['direction'])} | {f(v['median_norm_lead'])} | "
                     f"{f(v['one_seed_short'])} |")
    L.append("")

    # 4 cross-task
    L.append("## 4. Cross-task candidates\n")
    L.append("Shared candidate = precursor in all three tasks, same direction in all three, and "
             f"max/min of the three median normalized leads ≤ {f(a.max_lead_ratio)}. Specificity "
             "against controls is PENDING.\n")
    L.append("| Signal | Precursor in tasks | All three | Same direction | Lead ratio (max/min) | Status |")
    L.append("|---|---|---|---|---|---|")
    for c in cross:
        status = "shared candidate (specificity pending)" if c["candidate"] else "not a shared candidate"
        L.append(f"| `{sig_label(c['signal'])}` | {', '.join(c['tasks']) or 'none'} | {f(c['in_all'])} | "
                 f"{f(c['same_dir']) if c['in_all'] else '—'} | {f(c['ratio'])} | {status} |")
    two = [c for c in cross if c["n_tasks"] == 2]
    L.append("")
    L.append("Pre-registered S2 trigger (signal is a precursor in exactly 2 of 3 tasks): "
             + (", ".join(f"`{sig_label(c['signal'])}` ({', '.join(c['tasks'])})" for c in two)
                if two else "none") + ".")
    short = [(t, s) for (t, s), v in ver.items() if v["one_seed_short"]]
    L.append("Pre-registered “fails by one seed” cases: "
             + (", ".join(f"`{sig_label(s)}` in {t}" for t, s in short) if short else "none") + ".\n")

    # 5 positive control
    L.append(f"## 5. Positive control (`{POS_CONTROL}`)\n")
    L.append("| Task | Seed | t50 | Onset | Lead | Norm. lead | Min lead | Direction | Qualifies | Flag |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for x in sorted(pc, key=lambda x: (TASK_ORDER.index(x["task"]), x["seed"])):
        flag = "NO ONSET" if x["onset"] is None else ("lead below minimum" if not x["qualifies"] else "")
        L.append(f"| {x['task']} | {x['seed']} | {x['t50']} | {f(x['onset'])} | {f(x['lead'])} | "
                 f"{f(x['norm_lead'])} | {f(x['min_lead'])} | {dirs(x['direction'])} | "
                 f"{f(x['qualifies'])} | {flag} |")
    L.append("")

    # 6 spearman
    L.append("## 6. Spearman correlation: onset step vs t50 (descriptive only)\n")
    L.append("Across seeds with an onset. ρ is “—” when n < 3 or a variable is constant.\n")
    L.append("| Signal | Task | n | ρ |")
    L.append("|---|---|---|---|")
    for s in SIGNALS + [POS_CONTROL]:
        for t in TASK_ORDER:
            xs = [x for x in rows if x["task"] == t and x["signal"] == s and x["onset"] is not None]
            if s == POS_CONTROL and t not in POS_CONTROL_TASKS:
                continue
            rho = spearman([x["onset"] for x in xs], [x["t50"] for x in xs])
            L.append(f"| `{sig_label(s)}` | {t} | {len(xs)} | {'—' if rho is None else format(rho, '.3f')} |")
    L.append("")

    # 7 figures
    L.append("## 7. Figures\n")
    for t in TASK_ORDER:
        L.append(f"- {TASK_LABEL[t]}: [{rel(figs[t])}]({rel(figs[t])})")
    L.append("")
    L.append("Each panel: one line per seed, x = (step − t50)/t50, y = smoothed signal − m; "
             "dashed line at x = 0 (t50); dot at each seed's onset.\n")

    # 8 implementation choices
    ci = notes["cols"]
    L.append("## 8. Implementation choices and ambiguities\n")
    items = [
        "t50 is computed with `stays_above_step` from checkpoints_analysis/jump_detection.py on "
        "metrics.csv and checked against the per-run table of logs/run_results.md before any "
        "signal is used; all matched.",
        "L = `model.n_layers` from each run's config.json, so the final residual is "
        + "; ".join(f"{t}: L = {sorted({r['L'] for r in runs if r['task'] == t})} → "
                    f"`resid{runs_L(runs, t)}_*`" for t in TASK_ORDER) + ".",
        "Signal 5 averages, and signal 6 takes the max over, every column matching "
        "`attn_L<digits>H<digits>_entropy` / `_prev_token`, i.e. all heads in all layers: "
        + "; ".join(f"{t}: {len(ci[t]['entropy_cols'])} entropy, {len(ci[t]['prev_token_cols'])} "
                    f"prev_token columns" for t in TASK_ORDER if t in ci) + ".",
        "Signal 4 drops only the step-0 row (where `cka_prev` is NaN by construction); its series "
        "starts at the second checkpoint. Any other NaN would be kept and listed in section 9.",
        "Smoothing is a centered rolling median over 5 checkpoints with min_periods=1 (windows are "
        "truncated at the series ends; NaNs are skipped). Centering means each smoothed value uses "
        "the next 2 checkpoints. The pre-registration says only “rolling median over 5 "
        "checkpoints”; centering was specified in the analysis instructions.",
        "Reference window = checkpoints with step in [0.05·t50, 0.30·t50], both ends inclusive, "
        "evaluated on the smoothed series. m and MAD are taken over the smoothed values in this "
        "window. sigma = max(1.4826·MAD, 1e-6·|m| + 1e-12) (“floor” read as a lower bound).",
        "“Outside the band” means strictly below m − 3σ or strictly above m + 3σ; a value exactly "
        "on the band edge counts as inside. A NaN smoothed value would count as not outside.",
        "Onset candidates are checkpoints with 0.30·t50 < step ≤ t50. The onset is the earliest "
        "candidate from which every checkpoint up to and including t50 is outside the band. The "
        "pre-registration says no onset if the signal is inside the band “just before t50”; the "
        "analysis instructions say “at t50”. The literal instruction (at t50) is implemented. "
        "Because qualifying requires lead ≥ 2 checkpoint intervals, any run with an onset that "
        "qualifies is also outside the band at the checkpoint just before t50, so the two "
        "readings give the same qualification results.",
        "Direction = sign(smoothed value at t50 − m), recorded for every run including those "
        "without an onset; precursor direction uses qualifying seeds only.",
        "Checkpoint interval = median step spacing of signals.csv (spacings are listed in section "
        "9 if not uniform). Minimum lead = max(2·interval, 0.05·t50); lead equal to the minimum "
        "qualifies.",
        "Precursor: n_qualifying ≥ n_seeds − 1 and the qualifying seeds have one direction. "
        "The pre-registration's “direction agrees” is read as agreement among qualifying seeds.",
        "Shared candidate requires a precursor in all three tasks with one common direction and "
        "max/min of the three median normalized leads ≤ 3. The pre-registered specificity "
        "condition (controls) is not evaluated, so no overall Supports/Partial/Refutes verdict is "
        "given. “One seed short” and the S2 trigger are reported as facts only.",
        "The positive control uses the same onset rule, and qualifies under the same minimum-lead "
        "rule.",
        "Spearman ρ uses average ranks for ties and is computed only over seeds with an onset.",
        "Only the pre-registered defaults were run. No other signals, thresholds or parameters "
        "were tried.",
    ]
    L += [f"{i}. {s}" for i, s in enumerate(items, 1)]
    L.append("")

    # 9 unexpected
    L.append("## 9. Anything unexpected\n")
    un = []
    if notes["spacing"]:
        un.append("Non-uniform checkpoint spacing: " + "; ".join(notes["spacing"]))
    if notes["nan"]:
        un.append("NaN values in signals other than the step-0 cka_prev: " + "; ".join(notes["nan"]))
    if notes["t50_not_ckpt"]:
        un.append("t50 is not a checkpoint step in signals.csv: " + "; ".join(notes["t50_not_ckpt"]))
    if notes["few_ref"]:
        un.append("Reference window with fewer than 3 checkpoints: " + "; ".join(notes["few_ref"]))
    if notes["floored"]:
        un.append(f"sigma hit the floor (MAD ≈ 0) in {len(notes['floored'])} signal-runs: "
                  + "; ".join(notes["floored"]))
    if notes["first_candidate"]:
        un.append(f"Onset at the first checkpoint after the reference window (signal already outside "
                  f"the band when the onset search starts, so the onset is pinned at the window "
                  f"edge) in {len(notes['first_candidate'])} signal-runs: "
                  + "; ".join(notes["first_candidate"]))
    if pc_none or pc_q < len(pc):
        un.append(f"Positive control: {len(pc) - pc_q} of {len(pc)} runs do not qualify "
                  f"({pc_none} with no onset).")
    mixed = [(t, s) for (t, s), v in ver.items() if v["direction"] == "mixed"]
    if mixed:
        un.append("Qualifying seeds disagree in direction: "
                  + ", ".join(f"`{sig_label(s)}` in {t}" for t, s in mixed))
    L += [f"- {u}" for u in un] if un else ["Nothing flagged by the automatic checks."]
    L.append("")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    a = parse_args(argv)
    runs, _ = load_runs(a)
    if not runs:
        sys.exit("no runs found")
    mismatches, info = check_t50(runs, a)
    for m in info:
        print("note:", m)
    if mismatches:
        print("STOP: t50 differs from logs/run_results.md; nothing written.", file=sys.stderr)
        for m in mismatches:
            print("  " + m, file=sys.stderr)
        sys.exit(2)
    rows, notes = analyse(runs, a)
    ver = verdicts(rows, runs, a)
    cross = cross_task(ver, a)
    figs = make_figures(runs, a)
    write_report(runs, rows, notes, ver, cross, figs, a, None)
    print(a.out.resolve())


if __name__ == "__main__":
    main()

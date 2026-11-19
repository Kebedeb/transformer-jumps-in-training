"""S3/S4 stress-test analysis, as pre-registered in docs/hypothesis.md (Stress tests S3, S4,
including "Deviations and clarifications" 1-6).

    python -m checkpoints_analysis.stress_analysis

Runs (identified by config.json experiment_name and seed; *_INCOMPLETE ignored):
  S3 grokking_s3_2L64    (grokking, 2 layers, width 64)  seeds 201-203
  S4 induction_s4_lr1e3  (induction, lr 1e-3)             seeds 201-203
  S4 icl_s4_lr3e4        (ICL, lr 3e-4)                   seeds 201-203

Reuses the onset/qualification/verdict functions of
checkpoints_analysis/precursor_analysis.py unchanged, with its pre-registered
defaults. Main-config runs (batch 1 + batch 2) are re-analysed with the same
functions for the descriptive comparison. Writes only logs/stress_results.md and
logs/figures/stress_<experiment_name>.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from checkpoints_analysis import precursor_analysis as pa
from checkpoints_analysis.jump_detection import (
    TASKS as JD_TASKS, first_crossing_step, read_metrics, stays_above_step,
)
from checkpoints_analysis.sensitivity import parse_confirmation_t50

REPO_ROOT = Path(__file__).resolve().parent.parent
# experiment_name -> (task, stress test, description)
EXPERIMENTS = {
    "grokking_s3_2L64": ("grokking", "S3", "grokking, 2 layers, width 64"),
    "induction_s4_lr1e3": ("induction", "S4", "induction, learning rate 1e-3"),
    "icl_s4_lr3e4": ("icl", "S4", "ICL, learning rate 3e-4"),
}
STRESS_SEEDS = {201, 202, 203}
BATCH1, BATCH2 = {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}
SEED_COLORS = {201: "#2a78d6", 202: "#eb6834", 203: "#1baf7a"}
# Readings of "in all but one seed (4 of 5, or 3 of 4)" for n jump runs (section 6).
READINGS = {
    "A": ("all but one", lambda n: n - 1),
    "B": ("all seeds (the examples 4/5, 3/4 are ≥ 75%; n − 1 = 2 of 3 is below that)",
          lambda n: n),
}
f, dirs, lab = pa.f, pa.dirs, pa.sig_label
REFUSE = {"precursor_results.md", "sensitivity_results.md", "confirmation_results.md",
          "sensitivity_batch2.md", "run_results.md"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Pre-registered S3/S4 stress-test analysis.")
    ap.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "logs" / "stress_results.md")
    ap.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "logs" / "figures")
    ap.add_argument("--confirmation-report", type=Path,
                    default=REPO_ROOT / "logs" / "confirmation_results.md",
                    help="read only: batch-2 t50 cross-check")
    args, rest = ap.parse_known_args(argv)
    if rest:
        sys.exit(f"unknown arguments {rest}: analysis parameters are fixed at the "
                 "pre-registered defaults")
    args.pa = pa.parse_args([])
    if args.out.name in REFUSE:
        sys.exit("refusing to overwrite an earlier report")
    return args


# ----------------------------------------------------------------------------- inventory
def expected_steps(cfg):
    from train_transformer_helpers import CheckpointSettings, make_checkpoint_steps
    c = cfg["checkpoints"]
    s = CheckpointSettings(dense_until=int(c.get("dense_until", 1000)),
                           dense_every=int(c.get("dense_every", 50)),
                           sparse_every=int(c.get("sparse_every", 500)),
                           save_optimizer=bool(c.get("save_optimizer", False)))
    return make_checkpoint_steps(int(cfg["training"]["train_steps"]), s)


def inventory(args):
    inv, ignored = [], []
    for d in sorted(args.runs_dir.iterdir()):
        if not d.is_dir():
            continue
        exp_dir = next((e for e in EXPERIMENTS if d.name.endswith("-" + e)
                        or d.name.endswith("-" + e + "_INCOMPLETE")), None)
        if exp_dir is None:
            continue
        if d.name.endswith("_INCOMPLETE"):
            ignored.append(d.name)
            continue
        cfg = json.loads((d / "config.json").read_text())
        exp, seed, task = cfg.get("experiment_name"), int(cfg["seed"]), cfg["task"]["name"]
        problems = []
        if exp != exp_dir:
            problems.append(f"experiment_name {exp} ≠ directory {exp_dir}")
        if task != EXPERIMENTS[exp_dir][0]:
            problems.append(f"task {task} ≠ expected {EXPERIMENTS[exp_dir][0]}")
        if seed not in STRESS_SEEDS:
            problems.append(f"seed {seed} not in 201-203")
        metric = JD_TASKS[task][0]
        train_steps = int(cfg["training"]["train_steps"])
        exp_steps = expected_steps(cfg)
        ck = d / "checkpoints"
        files = [p for p in ck.iterdir() if p.is_file()] if ck.is_dir() else []
        empty = sum(p.stat().st_size == 0 for p in files)
        got_steps = sorted(int(p.stem.split("_")[-1]) for p in files if p.stem.startswith("ckpt_step_"))
        final, series = (read_metrics(d / "metrics.csv", metric)
                         if (d / "metrics.csv").is_file() else (None, []))
        if final is None or final < train_steps:
            problems.append(f"final step {final} < train_steps {train_steps}")
        if len(files) < len(exp_steps):
            problems.append(f"{len(files)} checkpoints < {len(exp_steps)}")
        missing = sorted(set(exp_steps) - set(got_steps))
        if missing:
            problems.append(f"missing checkpoint steps {missing[:5]}{'…' if len(missing) > 5 else ''}")
        if empty:
            problems.append(f"{empty} zero-byte checkpoint(s)")
        has_sig = (d / "signals.csv").is_file()
        if not has_sig:
            problems.append("no signals.csv")
        inv.append({"dir": d, "name": d.name, "exp": exp_dir, "task": task, "seed": seed,
                    "cfg": cfg, "metric": metric, "series": series, "final": final,
                    "train_steps": train_steps, "n_ckpt": len(files), "expected": len(exp_steps),
                    "has_signals": has_sig, "empty": empty, "problems": problems})
    return inv, ignored


def to_run(item, t50):
    return {"task": item["task"], "seed": item["seed"], "dir": item["name"], "config": item["cfg"],
            "L": int(item["cfg"]["model"]["n_layers"]), "t50": t50, "exp": item["exp"],
            "cols": pa.read_csv_columns(item["dir"] / "signals.csv")}


def by(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def lead_str(x):
    return f(x["norm_lead"]) + (" (WL)" if x["onset_at_first_candidate"] else "")


def signals_for(task):
    return pa.SIGNALS + ([pa.POS_CONTROL] if task in pa.POS_CONTROL_TASKS else [])


# ----------------------------------------------------------------------------- main configs
def load_main(args):
    """Batch 1 and batch 2 main runs, analysed per batch with the pre-registered defaults."""
    a = args.pa
    out = {}
    for name, seeds in (("B1", BATCH1), ("B2", BATCH2)):
        aa = pa.parse_args(["--seeds", ",".join(map(str, sorted(seeds)))])
        runs, _ = pa.load_runs(aa)
        if name == "B1":
            mism, _ = pa.check_t50(runs, a)
        else:
            ref = parse_confirmation_t50(args.confirmation_report)
            mism = [f"{r['task']} seed {r['seed']}: {r['t50']} vs {ref.get((r['task'], r['seed']))}"
                    for r in runs if ref.get((r["task"], r["seed"])) != r["t50"]]
        if mism:
            sys.exit(f"STOP: main-config {name} t50 check failed: {mism}")
        rows, _ = pa.analyse(runs, a)
        out[name] = {"runs": runs, "rows": rows, "ver": pa.verdicts(rows, runs, a)}
    return out


def stress_verdict(rows, n, need):
    q = [x for x in rows if x["qualifies"]]
    ds = sorted({x["direction"] for x in q})
    return {"n_qual": len(q), "n": n, "need": need,
            "precursor": n > 0 and len(q) >= need and len(ds) == 1,
            "direction": ds[0] if len(ds) == 1 else ("mixed" if ds else None),
            "median_norm_lead": float(np.median([x["norm_lead"] for x in q])) if q else None,
            "one_short": n > 0 and len(q) == need - 1}


# ----------------------------------------------------------------------------- figures
def make_figures(runs_by_exp, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for exp, (task, test, desc) in EXPERIMENTS.items():
        rs = sorted(runs_by_exp.get(exp, []), key=lambda r: r["seed"])
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
        L = rs[0]["L"] if rs else "L"
        for ax, sig in zip(axes.flat, pa.SIGNALS):
            for r in rs:
                steps, sm, res = r["smoothed"][sig]
                if res["m"] is None:
                    continue
                ax.plot((steps - r["t50"]) / r["t50"], sm - res["m"],
                        color=SEED_COLORS.get(r["seed"], "#6250d6"), lw=1.3, alpha=0.85, zorder=2)
            ax.axvline(0, color="#444444", lw=1, ls=":")
            ax.set_title(sig.replace("{L}", str(L)), fontsize=11)
            ax.set_xlabel("(step − t50) / t50")
            ax.set_ylabel("smoothed − reference median m")
            ax.grid(alpha=0.25, lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if not rs:
                ax.text(0.5, 0.5, "no run with a jump", ha="center", va="center",
                        transform=ax.transAxes, color="#777777")
        H = [plt.Line2D([], [], color=SEED_COLORS[s], lw=2) for s in sorted(STRESS_SEEDS)]
        labels = [f"seed {s}" + ("" if any(r["seed"] == s for r in rs) else " (no jump, not shown)")
                  for s in sorted(STRESS_SEEDS)]
        fig.legend(H, labels, loc="outside lower center", ncol=len(H), frameon=False)
        fig.suptitle(f"{test} {exp} ({desc}): candidate signals around t50 (dotted line = t50)",
                     fontsize=13, x=0.01, ha="left")
        p = args.fig_dir / f"stress_{exp}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths[exp] = p
    return paths


# ----------------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    a = args.pa
    inv, ignored = inventory(args)
    found = {(x["exp"], x["seed"]) for x in inv}
    absent = [(e, s) for e in EXPERIMENTS for s in sorted(STRESS_SEEDS) if (e, s) not in found]
    bad = [x for x in inv if x["problems"]]
    dup = len(found) != len(inv)
    if bad or absent or dup:
        print("STOP: incomplete, missing or unidentified stress runs; nothing written.", file=sys.stderr)
        for x in bad:
            print(f"  {x['name']} (seed {x['seed']}): {'; '.join(x['problems'])}", file=sys.stderr)
        for e, s in absent:
            print(f"  {e} seed {s}: no complete run directory", file=sys.stderr)
        if dup:
            print("  more than one directory for some (experiment, seed)", file=sys.stderr)
        sys.exit(2)

    # ---- behaviour (metrics.csv)
    notes = {"t50_check": [], "fc": []}
    for x in inv:
        x["t50"] = stays_above_step(x["series"], 0.5)
        x["t90"] = stays_above_step(x["series"], 0.9)
        x["fc50"] = first_crossing_step(x["series"], 0.5)
        x["final_metric"] = x["series"][-1][1] if x["series"] else None
        x["max_metric"] = max(v for _, v in x["series"]) if x["series"] else None
        if x["t50"] is not None:
            sc = pa.read_csv_columns(x["dir"] / "signals.csv")
            if x["metric"] in sc:
                ss = [(int(s), v) for s, v in zip(sc["step"], sc[x["metric"]]) if not np.isnan(v)]
                t = stays_above_step(ss, 0.5)
                if t != x["t50"]:
                    notes["t50_check"].append(f"{x['exp']} seed {x['seed']}: metrics.csv t50="
                                              f"{x['t50']}, signals.csv t50={t}")
            else:
                notes["t50_check"].append(f"{x['exp']} seed {x['seed']}: `{x['metric']}` not in "
                                          "signals.csv, cross-check not possible")
        if x["fc50"] is not None and x["fc50"] != x["t50"]:
            notes["fc"].append(f"{x['exp']} seed {x['seed']}: first crossing {x['fc50']}, "
                               f"stays-above t50 {f(x['t50'])}")

    # ---- signals (jump runs only)
    runs_by_exp, rows, anotes = {}, [], {}
    for exp in EXPERIMENTS:
        items = sorted((x for x in inv if x["exp"] == exp and x["t50"] is not None),
                       key=lambda x: x["seed"])
        rs = [to_run(x, x["t50"]) for x in items]
        runs_by_exp[exp] = rs
        if rs:
            rr, nt = pa.analyse(rs, a)
            for r in rr:
                r["exp"] = exp
            rows += rr
            anotes[exp] = nt

    # ---- verdicts under each reading
    ver = {}
    for exp, (task, _, _) in EXPERIMENTS.items():
        n = len(runs_by_exp[exp])
        for s in signals_for(task):
            rs = by(rows, exp=exp, signal=s)
            for R, (_, need_fn) in READINGS.items():
                ver[(exp, s, R)] = stress_verdict(rs, n, need_fn(n))

    main_cfg = load_main(args)
    figs = make_figures(runs_by_exp, args)
    write_report(args, a, inv, ignored, notes, anotes, runs_by_exp, rows, ver, main_cfg, figs)
    print(args.out.resolve())


# ----------------------------------------------------------------------------- report
def main_status(main_cfg, task, s):
    v1, v2 = main_cfg["B1"]["ver"][(task, s)], main_cfg["B2"]["ver"][(task, s)]
    if v1["precursor"] and v2["precursor"]:
        st = "precursor in both batches"
    elif v1["precursor"] or v2["precursor"]:
        st = f"precursor in {'B1' if v1['precursor'] else 'B2'} only"
    else:
        st = "precursor in neither batch"
    d = v1["direction"] if (v1["precursor"] and v2["precursor"]
                            and v1["direction"] == v2["direction"]) else None
    return st, d, v1, v2


def pooled(main_cfg, task, s):
    xs = by(main_cfg["B1"]["rows"], task=task, signal=s) + by(main_cfg["B2"]["rows"], task=task, signal=s)
    q = [x for x in xs if x["qualifies"]]
    npos = sum(x["direction"] > 0 for x in q)
    med = float(np.median([x["norm_lead"] for x in q])) if q else None
    return len(q), len(xs), npos, len(q) - npos, med


def compare(stress_prec, st, stress_dir, main_dir):
    if st == "precursor in both batches":
        status = "same" if stress_prec else "different"
    elif st == "precursor in neither batch":
        status = "different" if stress_prec else "same"
    else:
        status = "main split across batches"
    if stress_prec and main_dir is not None:
        d = "same" if stress_dir == main_dir else "different"
    else:
        d = "—"
    return status, d


def write_report(args, a, inv, ignored, notes, anotes, runs_by_exp, rows, ver, main_cfg, figs):
    L = []
    rel = lambda p: Path(p).resolve().relative_to(args.out.parent.resolve()).as_posix()  # noqa
    nojump = [x for x in inv if x["t50"] is None]

    # ---------------- 0 summary
    L.append("# Stress-test results: S3 and S4 (pre-registered)\n")
    L.append("Generated by `python -m checkpoints_analysis.stress_analysis`. Do not edit by hand; "
             "re-run the script.\n")
    L.append("## 0. Summary\n")
    L.append(f"- Runs: {len(inv)} (3 experiments × seeds 201-203); all complete. Runs with no jump "
             f"(never stay ≥ 0.5): " + (", ".join(f"{x['exp']} seed {x['seed']}" for x in nojump)
                                       or "none") + ".")
    for exp, (task, test, _) in EXPERIMENTS.items():
        n = len(runs_by_exp[exp])
        pa_ = [s for s in pa.SIGNALS if ver[(exp, s, "A")]["precursor"]]
        pb_ = [s for s in pa.SIGNALS if ver[(exp, s, "B")]["precursor"]]
        L.append(f"- {test} `{exp}` ({n} jump runs): precursor in {len(pa_)}/6 signals under reading A "
                 f"(≥ {n - 1} of {n}), {len(pb_)}/6 under reading B ({n} of {n})"
                 + (f"; reading B: {', '.join(f'`{lab(s)}`' for s in pb_)}" if pb_ else "") + ".")
    pc = by(rows, signal=pa.POS_CONTROL)
    L.append(f"- Positive control `{pa.POS_CONTROL}`: {sum(x['qualifies'] for x in pc)} of {len(pc)} "
             "induction/ICL stress runs qualify.")
    nsame = ndiff = 0
    for exp, (task, _, _) in EXPERIMENTS.items():
        for s in pa.SIGNALS:
            st, md, _, _ = main_status(main_cfg, task, s)
            for R in READINGS:
                v = ver[(exp, s, R)]
                c, _ = compare(v["precursor"], st, v["direction"], md)
                nsame += c == "same"
                ndiff += c == "different"
    L.append(f"- Precursor status vs main config (18 signal × experiment cells × 2 readings = 36): "
             f"same {nsame}, different {ndiff}, main split across batches {36 - nsame - ndiff}.")
    L.append("- Specificity not assessed for stress configs (controls exist for main configs only).")
    L.append("- The pre-registered precursor rule does not fix the threshold for 3 seeds; both "
             "readings are reported (section 6).\n")

    # ---------------- 1 provenance
    L.append("## 1. Provenance\n")
    L.append(f"- HEAD: `{pa.git('rev-parse', 'HEAD')}`")
    L.append("- Commits touching docs/hypothesis.md (newest first):")
    for line in pa.git("log", "--format=%H %ad %s", "--date=iso", "--", "docs/hypothesis.md").splitlines():
        L.append(f"  - `{line}`")
    dirty = pa.git("status", "--porcelain", "--", "docs/hypothesis.md", "checkpoints_analysis")
    L.append(f"- Uncommitted changes in docs/hypothesis.md or checkpoints_analysis/: "
             f"{'none' if not dirty else '`' + dirty.replace(chr(10), '; ') + '`'}")
    L.append(f"- Python {sys.version.split()[0]} (`{sys.executable}`), numpy {np.__version__}")
    L.append("- Analysis parameters (precursor_analysis defaults, unchanged): " + ", ".join(
        f"`{k}={getattr(a, k)}`" for k in pa.PARAM_KEYS))
    L.append("- Main-config comparison runs: batch 1 = " + ", ".join(
        f"{r['task']} s{r['seed']}" for r in main_cfg["B1"]["runs"]) + "; batch 2 = " + ", ".join(
        f"{r['task']} s{r['seed']}" for r in main_cfg["B2"]["runs"]) + ".")
    L.append("- Stress configs (from config.json):\n")
    L.append("| Experiment | Test | Task | n_layers | n_heads | n_embd | learning_rate | weight_decay "
             "| train_steps | eval_interval | checkpoint every |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for exp, (task, test, _) in EXPERIMENTS.items():
        c = next(x["cfg"] for x in inv if x["exp"] == exp)
        m, t, ck = c["model"], c["training"], c["checkpoints"]
        L.append(f"| `{exp}` | {test} | {task} | {m['n_layers']} | {m['n_heads']} | {m['n_embd']} | "
                 f"{t['learning_rate']} | {t['weight_decay']} | {t['train_steps']} | "
                 f"{t['eval_interval']} | {ck['dense_every']} |")
    L.append("")

    # ---------------- 2 inventory
    L.append("## 2. Inventory\n")
    L.append("| Run | Experiment | Seed | Final step | train_steps | Checkpoints | Expected | "
             "Zero-byte | signals.csv |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for x in sorted(inv, key=lambda x: (list(EXPERIMENTS).index(x["exp"]), x["seed"])):
        L.append(f"| {x['name']} | `{x['exp']}` | {x['seed']} | {x['final']} | {x['train_steps']} | "
                 f"{x['n_ckpt']} | {x['expected']} | {x['empty']} | {'yes' if x['has_signals'] else 'no'} |")
    L.append("")
    L.append("Expected checkpoints = `make_checkpoint_steps(train_steps, checkpoints)` from "
             "train_transformer_helpers.py; every expected step is present. "
             "Ignored `_INCOMPLETE` directories: " + (", ".join(f"`{n}`" for n in ignored) or "none")
             + ".\n")

    # ---------------- 3 per-run tables
    L.append("## 3. Per-run tables\n")
    L.append("### 3.1 Behaviour (metrics.csv)\n")
    L.append("| Experiment | Seed | Metric | t50 (stays ≥ 0.5) | First crossing 0.5 | t90 | Rise (t90 − t50) "
             "| Max metric | Final metric | Status |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for x in sorted(inv, key=lambda x: (list(EXPERIMENTS).index(x["exp"]), x["seed"])):
        rise = x["t90"] - x["t50"] if x["t90"] is not None and x["t50"] is not None else None
        L.append(f"| `{x['exp']}` | {x['seed']} | `{x['metric']}` | {f(x['t50'])} | {f(x['fc50'])} | "
                 f"{f(x['t90'])} | {f(rise)} | {x['max_metric']:.4f} | {x['final_metric']:.4f} | "
                 f"{'jump' if x['t50'] is not None else '**no jump** (signals not analysed)'} |")
    L.append("")
    L.append("### 3.2 Signals (jump runs only)\n")
    L.append("Lead = t50 − onset; norm. lead = lead / t50; WL = window-limited (onset at the first "
             "checkpoint after the reference window; normalized lead is a lower bound, deviation 2). "
             "Min lead = max(2 × checkpoint interval, 0.05 × t50). Direction = sign(smoothed signal "
             "at t50 − m). Lead < rise is an exploratory reading aid (deviation 3), not used for "
             "verdicts.\n")
    rise_of = {(x["exp"], x["seed"]): (x["t90"] - x["t50"] if x["t90"] is not None and x["t50"]
                                       is not None else None) for x in inv}
    for exp, (task, test, _) in EXPERIMENTS.items():
        Lr = runs_by_exp[exp][0]["L"] if runs_by_exp[exp] else "?"
        L.append(f"#### {test} `{exp}` (L = {Lr})\n")
        if not runs_by_exp[exp]:
            L.append("No run with a jump; nothing analysed.\n")
            continue
        L.append("| Signal | Seed | t50 | Onset | Lead | Norm. lead | Min lead | Direction | "
                 "Qualifies | Rise | Lead < rise |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for s in signals_for(task):
            for x in sorted(by(rows, exp=exp, signal=s), key=lambda x: x["seed"]):
                rz = rise_of[(exp, x["seed"])]
                lr = "—" if x["lead"] is None or rz is None else ("yes" if x["lead"] < rz else "no")
                L.append(f"| `{lab(s)}` | {x['seed']} | {x['t50']} | {f(x['onset'])} | {f(x['lead'])} | "
                         f"{lead_str(x) if x['lead'] is not None else '—'} | {f(x['min_lead'])} | "
                         f"{dirs(x['direction'])} | {f(x['qualifies'])} | {f(rz)} | {lr} |")
        L.append("")

    # ---------------- 4 per-experiment verdicts
    L.append("## 4. Per-experiment verdicts\n")
    L.append("n = number of jump runs. Reading A: precursor if ≥ n − 1 seeds qualify; reading B: "
             "all n seeds qualify. Both also require one direction among qualifying seeds "
             "(section 6). Specificity: **not assessed** for stress configs.\n")
    for exp, (task, test, _) in EXPERIMENTS.items():
        n = len(runs_by_exp[exp])
        L.append(f"### {test} `{exp}`\n")
        L.append("| Signal | Qualifying / n | Directions of qualifying (+/−) | Median norm. lead | "
                 "WL among qualifying | Precursor, reading A | Precursor, reading B | Specificity |")
        L.append("|---|---|---|---|---|---|---|---|")
        for s in signals_for(task):
            rs = by(rows, exp=exp, signal=s)
            q = [x for x in rs if x["qualifies"]]
            npos = sum(x["direction"] > 0 for x in q)
            wl = sum(x["onset_at_first_candidate"] for x in q)
            va, vb = ver[(exp, s, "A")], ver[(exp, s, "B")]
            cell = lambda v: (f"**yes** {dirs(v['direction'])}" if v["precursor"] else "no") + \
                f" (need {v['need']})"  # noqa: E731
            name = f"`{lab(s)}`" + (" (positive control)" if s == pa.POS_CONTROL else "")
            L.append(f"| {name} | {len(q)} / {n} | {npos}/{len(q) - npos} | {f(va['median_norm_lead'])} | "
                     f"{wl} | {cell(va)} | {cell(vb)} | not assessed |")
        L.append("")

    # ---------------- 5 comparison
    L.append("## 5. Comparison with main configs\n")
    L.append("Main config = the same task's `_main` runs, batch 1 (seeds 1-5; grokking 1-4) and "
             "batch 2 (seeds 6-10), each analysed with the pre-registered rule (n − 1 of n). "
             "“Main status” combines the two batch verdicts; the pooled columns (all main seeds) "
             "are descriptive only. Status comparison: same = stress precursor and main precursor "
             "in both batches, or neither; different = one but not the other; “main split” = "
             "main precursor in one batch only (no single main status). Direction is compared only "
             "where both the stress experiment and both main batches are precursors.\n")
    L.append("| Experiment | Signal | Main status | Main dir | Pooled qual / n | Pooled dirs (+/−) | "
             "Pooled median norm. lead | Stress A | Stress B | Stress dir | Status vs main (A) | "
             "Status vs main (B) | Direction vs main |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for exp, (task, test, _) in EXPERIMENTS.items():
        for s in pa.SIGNALS:
            st, md, v1, v2 = main_status(main_cfg, task, s)
            qn, nn, npos, nneg, med = pooled(main_cfg, task, s)
            va, vb = ver[(exp, s, "A")], ver[(exp, s, "B")]
            ca, da = compare(va["precursor"], st, va["direction"], md)
            cb, db = compare(vb["precursor"], st, vb["direction"], md)
            dcmp = da if da != "—" else db
            sdir = va["direction"] if va["precursor"] else None
            mx = lambda v: ", mixed dir" if v["direction"] == "mixed" else ""  # noqa: E731
            L.append(f"| `{exp}` | `{lab(s)}` | {st} (B1 {v1['n_qual']}/{v1['n_seeds']}{mx(v1)}, B2 "
                     f"{v2['n_qual']}/{v2['n_seeds']}{mx(v2)}) | {dirs(md)} | {qn}/{nn} | {npos}/{nneg} | "
                     f"{f(med)} | {f(va['precursor'])} | {f(vb['precursor'])} | {dirs(sdir)} | "
                     f"{ca} | {cb} | {dcmp} |")
    L.append("")
    L.append("Positive control (`attn_max_pattern`), descriptive:\n")
    L.append("| Experiment | Stress qual / n | Stress dirs (+/−) | Main pooled qual / n | Main pooled dirs (+/−) |")
    L.append("|---|---|---|---|---|")
    for exp, (task, _, _) in EXPERIMENTS.items():
        if task not in pa.POS_CONTROL_TASKS:
            continue
        q = [x for x in by(rows, exp=exp, signal=pa.POS_CONTROL) if x["qualifies"]]
        qn, nn, npos, nneg, _ = pooled(main_cfg, task, pa.POS_CONTROL)
        L.append(f"| `{exp}` | {len(q)}/{len(runs_by_exp[exp])} | "
                 f"{sum(x['direction'] > 0 for x in q)}/{sum(x['direction'] < 0 for x in q)} | "
                 f"{qn}/{nn} | {npos}/{nneg} |")
    L.append("")
    L.append("Final residual used: main grokking `resid1` (1 layer), S3 grokking `resid2` (2 layers); "
             "induction and ICL `resid2` in both main and S4 configs.\n")

    # ---------------- 6 ambiguities
    L.append("## 6. Rule ambiguities\n")
    amb = [
        "Precursor rule for 3 seeds. docs/hypothesis.md says “in all but one seed (4 of 5, or 3 of "
        "4)”. For n = 3 this can be read as (A) all but one = 2 of 3, or (B) all 3 of 3, because "
        "the two examples given both require at least 75% of seeds and 2 of 3 is 67%. Both readings "
        "are reported in sections 4 and 5; neither is chosen. If a run has no jump, n is the "
        "number of jump runs, so the readings become 1 of 2 and 2 of 2; whether a no-jump run should "
        "instead count as a failing seed out of 3 is also not specified.",
        "“Inconclusive: a signal fails by one seed” for n = 3: under reading A this is 1 of 3 "
        "qualifying, under reading B 2 of 3. Cells that are one seed short: reading A: "
        + (", ".join(f"`{lab(s)}` in {e}" for (e, s, R), v in ver.items() if R == "A"
                     and v["one_short"] and s in pa.SIGNALS) or "none")
        + "; reading B: "
        + (", ".join(f"`{lab(s)}` in {e}" for (e, s, R), v in ver.items() if R == "B"
                     and v["one_short"] and s in pa.SIGNALS) or "none") + ".",
        "The pre-registration lists checkpoint intervals of 10 for induction and ICL and 100 for "
        "grokking. The ICL S4 config checkpoints every 20 steps. The main analysis code takes the "
        "interval from the median spacing of signals.csv, so the ICL S4 minimum lead is "
        "max(40, 0.05 × t50). This affects the minimum lead only when 0.05 × t50 < 40, i.e. "
        "t50 < 800; ICL S4 t50 values: "
        + ", ".join(f"seed {x['seed']} {f(x['t50'])}" for x in inv if x["exp"] == "icl_s4_lr3e4")
        + ". No alternative interval was run.",
        "The pre-registration does not say how S3/S4 outcomes enter the decision rules (Supports / "
        "Partial / Refutes / Inconclusive). No overall label is derived here.",
        "The pre-registration does not define a comparison between stress and main configs. Section 5 "
        "uses “precursor in both batches” (deviation 1) as the main-config status; this choice is "
        "listed in section 7.",
    ]
    L += [f"{i}. {s}" for i, s in enumerate(amb, 1)]
    L.append("")

    # ---------------- 7 implementation choices
    L.append("## 7. Implementation choices\n")
    items = [
        "Runs are matched by directory suffix `-<experiment_name>` and verified against "
        "`experiment_name`, `task.name` and `seed` (201-203) in config.json; `_INCOMPLETE` "
        "directories are ignored. The script stops before analysis if any run is missing, "
        "incomplete (final step < train_steps, missing or zero-byte checkpoints) or lacks "
        "signals.csv.",
        "t50 = stays-above 0.5 on metrics.csv (`stays_above_step`, jump_detection.py), with the "
        "task's main metric (induction `heldout_acc`, ICL `fresh_map_acc`, grokking `val_acc`). "
        "It is cross-checked against the same column of signals.csv; disagreements are listed in "
        "section 8 and metrics.csv is used. Runs with no t50 are reported as “no jump” and their "
        "signals are not analysed.",
        "Layer convention, as in the main analysis: L = `model.n_layers` from config.json and the "
        "final residual is `resid{L}`. All three stress configs have 2 layers, so signals 2-4 use "
        "`resid2_*` (for S3 grokking this differs from main grokking, which uses `resid1_*`). "
        "Signal 5 averages and signal 6 maxes over every `attn_L<l>H<h>_*` column, i.e. all heads "
        "in both layers.",
        "Onset, lead, minimum lead, direction and qualification are computed by "
        "`precursor_analysis.analyse` with the pre-registered defaults (centered 5-checkpoint "
        "rolling median, reference window 5-30% of t50, 3 × 1.4826 × MAD band, minimum lead "
        "max(2 × interval, 0.05 × t50)). The checkpoint interval is the median step spacing of "
        "signals.csv (10 induction S4, 20 ICL S4, 100 grokking S3).",
        "Per-experiment verdicts use the same logic as `precursor_analysis.verdicts` (enough "
        "qualifying seeds and one direction among them) with the required count set by reading A "
        "(n − 1) or B (n); `verdicts` itself hard-codes n − 1, so the count is applied in this "
        "script.",
        "The positive control `attn_max_pattern` is analysed for induction and ICL only, with the "
        "same onset rule.",
        "Main-config comparison: batch 1 and batch 2 `_main` runs are loaded with "
        "`precursor_analysis.load_runs` and analysed separately with the defaults; batch-1 t50 is "
        "checked against logs/run_results.md and batch-2 t50 against section 3.1 of "
        "logs/confirmation_results.md. “Main dir” is shown only when both batches are precursors "
        "with the same direction.",
        "Specificity is not assessed: the C1/C2 controls were run on the main configs only.",
        "Figures follow the confirmation figures: 2 × 3 panels (the six signals), x = (step − t50)/t50, "
        "y = smoothed signal − the run's reference median m, dotted line at t50, one colour per "
        "seed. No-jump runs are not plotted (no t50).",
        "Only the pre-registered defaults were run; no other signals or parameters were tried.",
    ]
    L += [f"{i}. {s}" for i, s in enumerate(items, 1)]
    L.append("")
    L.append("Figures:\n")
    for exp in EXPERIMENTS:
        L.append(f"- `{exp}`: [{rel(figs[exp])}]({rel(figs[exp])})")
    L.append("")

    # ---------------- 8 unexpected
    L.append("## 8. Anything unexpected\n")
    un = []
    if nojump:
        un.append("Runs that never stay ≥ 0.5 (no jump): " + "; ".join(
            f"{x['exp']} seed {x['seed']} (max {x['metric']} {x['max_metric']:.4f}, final "
            f"{x['final_metric']:.4f}, first crossing {f(x['fc50'])})" for x in nojump))
    if notes["t50_check"]:
        un.append("t50 on metrics.csv vs signals.csv: " + "; ".join(notes["t50_check"]))
    if notes["fc"]:
        un.append("First crossing of 0.5 differs from stays-above t50: " + "; ".join(notes["fc"]))
    for exp, nt in anotes.items():
        if nt["spacing"]:
            un.append(f"{exp}: non-uniform checkpoint spacing: " + "; ".join(nt["spacing"]))
        if nt["nan"]:
            un.append(f"{exp}: NaNs beyond step-0 cka_prev: " + "; ".join(nt["nan"]))
        if nt["t50_not_ckpt"]:
            un.append(f"{exp}: t50 not a checkpoint step: " + "; ".join(nt["t50_not_ckpt"]))
        if nt["floored"]:
            un.append(f"{exp}: sigma at floor in {len(nt['floored'])} signal-runs: " + "; ".join(nt["floored"]))
        if nt["few_ref"]:
            un.append(f"{exp}: reference window with < 3 checkpoints: " + "; ".join(nt["few_ref"]))
    for exp in EXPERIMENTS:
        nr = sorted({x["n_ref"] for x in by(rows, exp=exp)})
        if nr and min(nr) <= 5:
            un.append(f"{exp}: reference window (5-30% of t50) contains {'/'.join(map(str, nr))} "
                      "checkpoints per signal-run.")
    wl = [x for x in rows if x["onset_at_first_candidate"]]
    if wl:
        un.append(f"{len(wl)} window-limited onsets: " + "; ".join(
            f"{x['exp']} seed {x['seed']} `{lab(x['signal'])}`" for x in wl))
    pcn = [x for x in by(rows, signal=pa.POS_CONTROL) if not x["qualifies"]]
    if pcn:
        un.append("Positive control not qualifying: " + "; ".join(
            f"{x['exp']} seed {x['seed']} (onset {f(x['onset'])}, lead {f(x['lead'])}, "
            f"min lead {f(x['min_lead'])})" for x in pcn))
    mixed = sorted({(e, s) for (e, s, R), v in ver.items() if v["direction"] == "mixed"})
    if mixed:
        un.append("Qualifying seeds disagree in direction: "
                  + ", ".join(f"`{lab(s)}` in {e}" for e, s in mixed))
    L += [f"- {u}" for u in un] if un else ["Nothing flagged by the automatic checks."]
    L.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

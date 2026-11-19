"""Confirmation (S2, batch 2) and control analysis, as pre-registered in docs/hypothesis.md
(including "Deviations and clarifications" 1-6).

    python -m checkpoints_analysis.confirmation_analysis

Reuses the onset/qualification/verdict functions of
checkpoints_analysis/precursor_analysis.py unchanged, with its pre-registered
defaults. Writes only logs/confirmation_results.md and
logs/figures/confirmation_<task>.png.

Runs (batch identified by config.json seed; *_INCOMPLETE ignored):
  batch 1  = runs/*-<task>_main, seeds 1-5 (grokking 1-4)
  batch 2  = runs/*-<task>_main, seeds 6-10
  controls = runs/*_ctrl_*, seeds 101-103
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from checkpoints_analysis import precursor_analysis as pa
from checkpoints_analysis.jump_detection import (
    TASKS as JD_TASKS, first_crossing_step, read_metrics, stays_above_step,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BATCH1 = {1, 2, 3, 4, 5}
BATCH2 = {6, 7, 8, 9, 10}
CONTROL_SEEDS = {101, 102, 103}
# experiment_name -> (task, control type). C1 natural no-jump, C2 shuffled targets.
CONTROLS = {
    "grokking_ctrl_wd0": ("grokking", "C1"),
    "icl_ctrl_fixedonly": ("icl", "C1"),
    "induction_ctrl_shuffle": ("induction", "C2"),
    "icl_ctrl_shuffle": ("icl", "C2"),
    "grokking_ctrl_shuffle": ("grokking", "C2"),
}
CTRL_TYPES_BY_TASK = {t: sorted({ct for (tt, ct) in CONTROLS.values() if tt == t})
                      for t in pa.TASK_ORDER}
COL_B1, COL_B2, COL_CTRL = "#2a78d6", "#eb6834", "#8a8a8a"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Pre-registered confirmation + control analysis.")
    ap.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "logs" / "confirmation_results.md")
    ap.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "logs" / "figures")
    ap.add_argument("--sensitivity-report", type=Path,
                    default=REPO_ROOT / "logs" / "sensitivity_results.md",
                    help="read only, for the 'flips across the sensitivity grid' rule")
    ap.add_argument("--anchors", default="induction=296,icl=816,grokking=7525",
                    help="pre-registered control anchors (batch-1 mean t50, deviation 5); "
                         "checked against the recomputed batch-1 mean")
    args, rest = ap.parse_known_args(argv)
    args.pa = pa.parse_args(rest)  # analysis parameters: pre-registered defaults
    args.anchors = {k: float(v) for k, v in (x.split("=") for x in args.anchors.split(","))}
    for out in (args.out, args.fig_dir / "x"):
        if out.name in ("precursor_results.md", "sensitivity_results.md"):
            sys.exit("refusing to overwrite an earlier report")
    return args


# ----------------------------------------------------------------------------- loading
def inventory(args):
    inv, ignored = [], []
    for d in sorted(args.runs_dir.iterdir()):
        if not d.is_dir():
            continue
        is_main = d.name.endswith("_main")
        is_ctrl = "_ctrl_" in d.name and not d.name.endswith("_INCOMPLETE")
        if not (is_main or is_ctrl):
            if d.name.endswith("_INCOMPLETE") or not d.name.endswith("_v1"):
                ignored.append(d.name)
            continue
        cfg = json.loads((d / "config.json").read_text())
        task = cfg["task"]["name"]
        seed = int(cfg["seed"])
        if is_main:
            kind = "batch1" if seed in BATCH1 else ("batch2" if seed in BATCH2 else "unknown seed")
            ctype = None
        else:
            exp = cfg.get("experiment_name")
            kind = "control" if exp in CONTROLS and seed in CONTROL_SEEDS else "unknown control"
            ctype = CONTROLS.get(exp, (None, None))[1]
        metric, expected = JD_TASKS[task]
        train_steps = int(cfg["training"]["train_steps"])
        ck = d / "checkpoints"
        files = [p for p in ck.iterdir() if p.is_file()] if ck.is_dir() else []
        empty = sum(p.stat().st_size == 0 for p in files)
        final, series = (read_metrics(d / "metrics.csv", metric)
                         if (d / "metrics.csv").is_file() else (None, []))
        problems = []
        if final is None or final < train_steps:
            problems.append(f"final step {final} < train_steps {train_steps}")
        if len(files) < expected:
            problems.append(f"{len(files)} checkpoints < {expected}")
        if empty:
            problems.append(f"{empty} zero-byte checkpoint(s)")
        if not (d / "signals.csv").is_file():
            problems.append("no signals.csv")
        if kind.startswith("unknown"):
            problems.append(kind)
        inv.append({"dir": d, "name": d.name, "task": task, "seed": seed, "kind": kind,
                    "ctype": ctype, "cfg": cfg, "metric": metric, "series": series,
                    "final": final, "train_steps": train_steps, "n_ckpt": len(files),
                    "expected": expected, "has_signals": (d / "signals.csv").is_file(),
                    "problems": problems})
    return inv, ignored


def to_run(item, t50):
    return {"task": item["task"], "seed": item["seed"], "dir": item["name"], "config": item["cfg"],
            "L": int(item["cfg"]["model"]["n_layers"]), "t50": t50,
            "cols": pa.read_csv_columns(item["dir"] / "signals.csv"),
            "kind": item["kind"], "ctype": item["ctype"]}


# ----------------------------------------------------------------------------- helpers
def by(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def one(rows, **kw):
    x = by(rows, **kw)
    assert len(x) == 1, (kw, len(x))
    return x[0]


f, dirs, lab = pa.f, pa.dirs, pa.sig_label


def lead_str(x):
    return f(x["norm_lead"]) + (" (WL)" if x["onset_at_first_candidate"] else "")


# ----------------------------------------------------------------------------- figures
def make_figures(jump_runs, ctrl_runs, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for task in pa.TASK_ORDER:
        jr = [r for r in jump_runs if r["task"] == task]
        cr = [r for r in ctrl_runs if r["task"] == task]
        L = jr[0]["L"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
        for ax, sig in zip(axes.flat, pa.SIGNALS):
            for r in cr:
                steps, sm, res = r["smoothed"][sig]
                if res["m"] is None:
                    continue
                ax.plot((steps - r["t50"]) / r["t50"], sm - res["m"], color=COL_CTRL, lw=1.1,
                        ls="-" if r["ctype"] == "C1" else "--", alpha=0.9, zorder=1)
            for r in jr:
                steps, sm, res = r["smoothed"][sig]
                if res["m"] is None:
                    continue
                c = COL_B1 if r["kind"] == "batch1" else COL_B2
                ax.plot((steps - r["t50"]) / r["t50"], sm - res["m"], color=c, lw=1.3,
                        alpha=0.85, zorder=2)
            ax.axvline(0, color="#444444", lw=1, ls=":")
            ax.set_title(sig.replace("{L}", str(L)), fontsize=11)
            ax.set_xlabel("(step − anchor) / anchor   [anchor = own t50 for jump runs]")
            ax.set_ylabel("smoothed − reference median m")
            ax.grid(alpha=0.25, lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        H = [plt.Line2D([], [], color=COL_B1, lw=2), plt.Line2D([], [], color=COL_B2, lw=2)]
        labels = ["batch 1 (seeds 1–5 / 1–4)", "batch 2 (seeds 6–10)"]
        for ct in CTRL_TYPES_BY_TASK[task]:
            H.append(plt.Line2D([], [], color=COL_CTRL, lw=2, ls="-" if ct == "C1" else "--"))
            labels.append(f"{ct} control (valid, seeds 101–103)")
        fig.legend(H, labels, loc="outside lower center", ncol=len(H), frameon=False)
        anc = cr[0]["t50"] if cr else None
        fig.suptitle(f"{pa.TASK_LABEL[task]}: jump runs vs controls (dotted line = t50 / "
                     f"control anchor {f(anc)})", fontsize=13, x=0.01, ha="left")
        p = args.fig_dir / f"confirmation_{task}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths[task] = p
    return paths


# ----------------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    a = args.pa
    inv, ignored = inventory(args)
    bad = [x for x in inv if x["problems"]]
    if bad:
        print("STOP: incomplete or unidentified runs; nothing written.", file=sys.stderr)
        for x in bad:
            print(f"  {x['name']} (seed {x['seed']}): {'; '.join(x['problems'])}", file=sys.stderr)
        sys.exit(2)

    # ---- t50 for jump runs (metrics.csv via jump_detection) + cross-check
    notes = {"t50_check": [], "anchor": []}
    jump_items = [x for x in inv if x["kind"] in ("batch1", "batch2")]
    for x in jump_items:
        x["t50"] = stays_above_step(x["series"], 0.5)
        x["t90"] = stays_above_step(x["series"], 0.9)
        x["fc50"] = first_crossing_step(x["series"], 0.5)
        sc = pa.read_csv_columns(x["dir"] / "signals.csv")
        sig_series = [(int(s), v) for s, v in zip(sc["step"], sc[x["metric"]]) if not np.isnan(v)]
        x["t50_signals"] = stays_above_step(sig_series, 0.5)
        if x["t50_signals"] != x["t50"]:
            notes["t50_check"].append(f"{x['task']} seed {x['seed']}: metrics.csv t50={x['t50']}, "
                                      f"signals.csv {x['metric']} t50={x['t50_signals']}")
    # batch 1 must also match logs/run_results.md (as in precursor_analysis)
    b1_runs = [to_run(x, x["t50"]) for x in jump_items if x["kind"] == "batch1"]
    mism, _ = pa.check_t50(b1_runs, a)
    if mism:
        print("STOP: batch-1 t50 differs from logs/run_results.md:", *mism, sep="\n  ", file=sys.stderr)
        sys.exit(2)
    b2_runs = [to_run(x, x["t50"]) for x in jump_items if x["kind"] == "batch2"]
    if any(r["t50"] is None for r in b2_runs):
        sys.exit("STOP: a batch-2 run never stays >= 0.5; t50 undefined")

    # ---- anchors
    anchors = {}
    for t in pa.TASK_ORDER:
        mean_t50 = float(np.mean([r["t50"] for r in b1_runs if r["task"] == t]))
        if abs(mean_t50 - args.anchors[t]) > 1e-9:
            sys.exit(f"STOP: batch-1 mean t50 for {t} is {mean_t50}, pre-registered anchor "
                     f"{args.anchors[t]}")
        anchors[t] = {"mean": mean_t50}

    # ---- controls: validity, snapped anchor
    ctrl_items = [x for x in inv if x["kind"] == "control"]
    for x in ctrl_items:
        x["stays"] = stays_above_step(x["series"], 0.5)
        x["ever"] = first_crossing_step(x["series"], 0.5)
        x["max_metric"] = max(v for _, v in x["series"])
        x["valid"] = x["stays"] is None
    ctrl_runs = []
    for x in ctrl_items:
        steps = pa.read_csv_columns(x["dir"] / "signals.csv")["step"]
        mean_t50 = anchors[x["task"]]["mean"]
        d = np.abs(steps - mean_t50)
        if np.sum(d == d.min()) > 1:
            notes["anchor"].append(f"{x['name']}: tie when snapping {mean_t50}; earlier step used")
        snapped = int(steps[np.argmin(d)])
        prev = anchors[x["task"]].get("snapped")
        if prev is not None and prev != snapped:
            notes["anchor"].append(f"{x['task']}: snapped anchor differs across controls "
                                   f"({prev} vs {snapped})")
        anchors[x["task"]]["snapped"] = snapped
        if x["valid"]:
            ctrl_runs.append(to_run(x, snapped))

    # ---- analyses (pre-registered functions, unchanged)
    rows1, notes1 = pa.analyse(b1_runs, a)
    rows2, notes2 = pa.analyse(b2_runs, a)
    # notes over all valid controls; rows per control run so each row carries its type
    _, notesc = pa.analyse(ctrl_runs, a) if ctrl_runs else ([], {})
    rowsc = []
    for c in ctrl_runs:
        rr, _ = pa.analyse([c], a)
        for r in rr:
            r["ctype"] = c["ctype"]
            r["dir"] = c["dir"]
        rowsc += rr
    ver1, ver2 = pa.verdicts(rows1, b1_runs, a), pa.verdicts(rows2, b2_runs, a)
    cross2 = pa.cross_task(ver2, a)
    # rise times for the exploratory reading aid (deviation 3)
    rise = {(x["task"], x["seed"]): (x["t90"] - x["t50"]) if x["t90"] is not None else None
            for x in jump_items}

    # ---- replication / confirmation
    rep = {}
    for t in pa.TASK_ORDER:
        for s in pa.SIGNALS:
            v1, v2 = ver1[(t, s)], ver2[(t, s)]
            rep[(t, s)] = {"v1": v1, "v2": v2, "confirmed": v1["precursor"] and v2["precursor"],
                           "same_dir": v1["direction"] == v2["direction"]}

    # ---- specificity
    spec = {}
    for t in pa.TASK_ORDER:
        for s in pa.SIGNALS:
            per = {}
            for ct in CTRL_TYPES_BY_TASK[t]:
                n_total = len([x for x in ctrl_items if x["task"] == t and x["ctype"] == ct])
                rs = by(rowsc, task=t, signal=s, ctype=ct)
                k = sum(r["qualifies"] for r in rs)
                per[ct] = {"k": k, "n_valid": len(rs), "n_total": n_total, "nonspec": k >= 2,
                           "dirs": [r["direction"] for r in sorted(rs, key=lambda r: r["seed"])]}
            nonspec = any(p["nonspec"] for p in per.values())
            conf = rep[(t, s)]["confirmed"]
            status = ("confirmed and specific" if conf and not nonspec else
                      "confirmed but non-specific" if conf else "not confirmed")
            spec[(t, s)] = {"per": per, "nonspec": nonspec, "status": status}

    # ---- cross-task on confirmed & specific signals (batch 2 values, batch 1 alongside)
    cross = []
    for s in pa.SIGNALS:
        ok_tasks = [t for t in pa.TASK_ORDER if spec[(t, s)]["status"] == "confirmed and specific"]
        eligible = len(ok_tasks) == 3
        d2 = {ver2[(t, s)]["direction"] for t in pa.TASK_ORDER}
        l2 = [ver2[(t, s)]["median_norm_lead"] for t in pa.TASK_ORDER]
        l1 = [ver1[(t, s)]["median_norm_lead"] for t in pa.TASK_ORDER]
        ratio = lambda ls: (max(ls) / min(ls)) if all(v is not None and v > 0 for v in ls) else None  # noqa
        same = eligible and len(d2) == 1
        r2 = ratio(l2) if eligible else None
        cross.append({"signal": s, "ok_tasks": ok_tasks, "eligible": eligible, "same_dir": same,
                      "ratio2": r2, "ratio1": ratio(l1),
                      "shared": eligible and same and r2 is not None and r2 <= a.max_lead_ratio})
    shared = [c["signal"] for c in cross if c["shared"]]

    # ---- outcome facts
    def tasks_qualifying(s, reading):
        if reading == "A":
            return [t for t in pa.TASK_ORDER if spec[(t, s)]["status"] == "confirmed and specific"]
        return [t for t in pa.TASK_ORDER if rep[(t, s)]["confirmed"]]
    readings = {"A": "“qualifies” = confirmed (precursor in both batches) and specific",
                "B": "“qualifies” = confirmed (precursor in both batches), specificity not considered"}
    partial, refutes1 = {}, {}
    for R in readings:
        q = {s: tasks_qualifying(s, R) for s in pa.SIGNALS}
        partial[R] = [(s, q[s]) for s in pa.SIGNALS if len(q[s]) == 2]
        refutes1[R] = not any(len(q[s]) >= 2 for s in pa.SIGNALS)
    confirmed_cells = [(t, s) for (t, s), v in rep.items() if v["confirmed"]]
    all_nonspec = bool(confirmed_cells) and all(spec[k]["nonspec"] for k in confirmed_cells)
    short1 = [(t, s) for (t, s), v in ver1.items() if v["one_seed_short"]]
    short2 = [(t, s) for (t, s), v in ver2.items() if v["one_seed_short"]]
    ctrl_ever = [x for x in ctrl_items if x["ever"] is not None]
    ctrl_stay = [x for x in ctrl_items if x["stays"] is not None]
    sens = parse_sensitivity(args.sensitivity_report)

    figs = make_figures(b1_runs + b2_runs, ctrl_runs, args)

    write_report(args, a, inv, ignored, notes, anchors, ctrl_items, b1_runs, b2_runs, ctrl_runs,
                 rows1, rows2, rowsc, ver1, ver2, cross2, rep, spec, cross, shared, rise,
                 readings, partial, refutes1, confirmed_cells, all_nonspec, short1, short2,
                 ctrl_ever, ctrl_stay, sens, figs, notes1, notes2, notesc)
    print(args.out.resolve())


def parse_sensitivity(path: Path):
    if not path.is_file():
        return None
    txt = path.read_text(encoding="utf-8")
    m = re.search(r"Per-task verdicts that change in at least one variant: (\d+) of (\d+)", txt)
    cand = re.search(r"Shared candidate \(specificity pending\) in any variant: ([^\n]*)", txt)
    return {"changed": int(m.group(1)) if m else None, "total": int(m.group(2)) if m else None,
            "shared_line": cand.group(1).strip() if cand else None}


# ----------------------------------------------------------------------------- report
def write_report(args, a, inv, ignored, notes, anchors, ctrl_items, b1_runs, b2_runs, ctrl_runs,
                 rows1, rows2, rowsc, ver1, ver2, cross2, rep, spec, cross, shared, rise,
                 readings, partial, refutes1, confirmed_cells, all_nonspec, short1, short2,
                 ctrl_ever, ctrl_stay, sens, figs, notes1, notes2, notesc):
    L = []
    rel = lambda p: Path(p).resolve().relative_to(args.out.parent.resolve()).as_posix()  # noqa
    n_conf = sum(v["confirmed"] for v in rep.values())
    n_cs = sum(v["status"] == "confirmed and specific" for v in spec.values())
    n_valid = sum(x["valid"] for x in ctrl_items)

    # rule outcomes
    applies = {}
    for R in readings:
        labels = []
        if shared:
            labels.append("Supports")
        if partial[R]:
            labels.append("Partial")
        if refutes1[R] or all_nonspec:
            labels.append("Refutes")
        applies[R] = labels
    inconc = []
    if short1 or short2:
        inconc.append("a signal fails by one seed")
    if ctrl_ever:
        inconc.append("a control reaches 0.5 at some evaluation")
    if sens and sens["changed"]:
        inconc.append("per-task verdicts flip across the S1 grid (batch 1)")

    L.append("# Confirmation and control results (pre-registered)\n")
    L.append("Generated by `python -m checkpoints_analysis.confirmation_analysis`. Do not edit by "
             "hand; re-run the script.\n")

    # 0 summary
    L.append("## 0. Summary\n")
    L.append(f"- Runs: batch 1 {len(b1_runs)}, batch 2 {len(b2_runs)}, controls {len(ctrl_items)} "
             f"({n_valid} valid); all complete.")
    L.append("- Batch-2 precursors: " + "; ".join(
        f"{t} {sum(ver2[(t, s)]['precursor'] for s in pa.SIGNALS)}/6" for t in pa.TASK_ORDER) + ".")
    L.append(f"- Confirmed (precursor in both batches): {n_conf} of 18 signal × task cells; "
             f"confirmed and specific: {n_cs}.")
    L.append("- Shared precursor (confirmed, specific, all three tasks, same direction, lead ratio ≤ "
             f"{f(a.max_lead_ratio)}): " + (", ".join(f"`{lab(s)}`" for s in shared) or "none") + ".")
    L.append(f"- Positive control `{pa.POS_CONTROL}` in batch 2: "
             f"{sum(r['qualifies'] for r in by(rows2, signal=pa.POS_CONTROL))} of "
             f"{len(by(rows2, signal=pa.POS_CONTROL))} induction/ICL runs qualify.")
    L.append("- Decision-rule labels whose conditions are met: " + "; ".join(
        f"reading {R}: {', '.join(applies[R]) or 'none of Supports/Partial/Refutes'}" for R in readings)
        + (f"; Inconclusive conditions met: {', '.join(inconc)}" if inconc else "") + ".")
    L.append("")

    # 1 provenance
    L.append("## 1. Provenance\n")
    L.append(f"- HEAD: `{pa.git('rev-parse', 'HEAD')}`")
    L.append("- Commits touching docs/hypothesis.md (newest first):")
    for line in pa.git("log", "--format=%H %ad %s", "--date=iso", "--", "docs/hypothesis.md").splitlines():
        L.append(f"  - `{line}`")
    L.append(f"- Last commit touching checkpoints_analysis/precursor_analysis.py: "
             f"`{pa.git('log', '-1', '--format=%H %ad', '--date=iso', '--', 'checkpoints_analysis/precursor_analysis.py')}`")
    dirty = pa.git("status", "--porcelain", "--", "docs/hypothesis.md", "checkpoints_analysis")
    L.append(f"- Uncommitted changes in docs/hypothesis.md or checkpoints_analysis/: "
             f"{'none' if not dirty else '`' + dirty.replace(chr(10), '; ') + '`'}")
    stamps = lambda kind: sorted(x["name"][:17] for x in inv if x["kind"] == kind)  # noqa
    for kind in ("batch2", "control"):
        s = stamps(kind)
        L.append(f"- {kind} run directory timestamps: {s[0]} … {s[-1]}")
    L.append(f"- Python {sys.version.split()[0]}, numpy {np.__version__}")
    L.append("- Analysis parameters (precursor_analysis defaults, unchanged): " + ", ".join(
        f"`{k}={getattr(a, k)}`" for k in pa.PARAM_KEYS))
    L.append("- Control anchors: " + ", ".join(
        f"{t} mean t50 {f(anchors[t]['mean'])} → snapped {anchors[t].get('snapped')}"
        for t in pa.TASK_ORDER))
    L.append("")

    # 2 inventory
    L.append("## 2. Run inventory and control validity\n")
    L.append("| Run | Task | Kind | Seed | Final step | train_steps | Checkpoints | Expected | signals.csv |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    order = {"batch1": 0, "batch2": 1, "control": 2}
    for x in sorted(inv, key=lambda x: (order.get(x["kind"], 3), pa.TASK_ORDER.index(x["task"]),
                                        x["ctype"] or "", x["seed"])):
        kind = x["kind"] + (f" {x['ctype']} ({x['cfg']['experiment_name']})" if x["ctype"] else "")
        L.append(f"| {x['name']} | {x['task']} | {kind} | {x['seed']} | {x['final']} | "
                 f"{x['train_steps']} | {x['n_ckpt']} | {x['expected']} | "
                 f"{'yes' if x['has_signals'] else 'no'} |")
    L.append("")
    L.append("Not matched (ignored): " + (", ".join(f"`{n}`" for n in ignored) or "none") + ".\n")
    L.append("Control validity (deviation 4: valid only if the main metric never stays ≥ 0.5):\n")
    L.append("| Run | Task | Type | Seed | Metric | Max metric | First step ≥ 0.5 | Stays ≥ 0.5 from | Status |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for x in sorted(ctrl_items, key=lambda x: (pa.TASK_ORDER.index(x["task"]), x["ctype"], x["seed"])):
        L.append(f"| {x['name']} | {x['task']} | {x['ctype']} | {x['seed']} | `{x['metric']}` | "
                 f"{x['max_metric']:.4f} | {f(x['ever'])} | {f(x['stays'])} | "
                 f"{'valid' if x['valid'] else 'INVALID (reaches 0.5 and stays)'} |")
    L.append("")

    # 3 batch 2
    L.append("## 3. Batch-2 results (seeds 6-10, analysed on their own)\n")
    L.append("### 3.1 Per-run table\n")
    L.append("WL = window-limited (onset at the first checkpoint after the reference window; the "
             "normalized lead is a lower bound, deviation 2). Rise = t90 − t50; “lead < rise” is "
             "an exploratory reading aid, not used for verdicts (deviation 3).\n")
    L.append("| Task | Signal | Seed | t50 | Onset | Lead | Norm. lead | Min lead | Direction | "
             "Qualifies | Rise | Lead < rise |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in pa.TASK_ORDER:
        for s in pa.SIGNALS:
            for x in sorted(by(rows2, task=t, signal=s), key=lambda x: x["seed"]):
                rz = rise[(t, x["seed"])]
                lr = "—" if x["lead"] is None or rz is None else ("yes" if x["lead"] < rz else "no")
                L.append(f"| {t} | `{lab(s)}` | {x['seed']} | {x['t50']} | {f(x['onset'])} | "
                         f"{f(x['lead'])} | {lead_str(x) if x['lead'] is not None else '—'} | "
                         f"{f(x['min_lead'])} | {dirs(x['direction'])} | {f(x['qualifies'])} | "
                         f"{f(rz)} | {lr} |")
    L.append("")
    L.append("### 3.2 Per-task verdict (batch 2)\n")
    L.append("| Signal | Task | Precursor | Qualifying / seeds | Direction | Median norm. lead | "
             "WL among qualifying | One seed short |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in pa.SIGNALS:
        for t in pa.TASK_ORDER:
            v = ver2[(t, s)]
            wl = sum(x["onset_at_first_candidate"] for x in by(rows2, task=t, signal=s) if x["qualifies"])
            L.append(f"| `{lab(s)}` | {t} | {f(v['precursor'])} | {v['n_qual']} / {v['n_seeds']} "
                     f"(need {v['need']}) | {dirs(v['direction'])} | {f(v['median_norm_lead'])} | "
                     f"{wl} | {f(v['one_seed_short'])} |")
    L.append("")
    L.append("### 3.3 Shared-candidate rule on batch 2 alone (all six signals, before confirmation "
             "and specificity)\n")
    L.append("| Signal | Precursor in tasks | All three | Same direction | Lead ratio | Meets rule |")
    L.append("|---|---|---|---|---|---|")
    for c in cross2:
        L.append(f"| `{lab(c['signal'])}` | {', '.join(c['tasks']) or 'none'} | {f(c['in_all'])} | "
                 f"{f(c['same_dir']) if c['in_all'] else '—'} | {f(c['ratio'])} | {f(c['candidate'])} |")
    L.append("")
    L.append(f"### 3.4 Positive control (`{pa.POS_CONTROL}`, batch 2)\n")
    L.append("| Task | Seed | t50 | Onset | Lead | Norm. lead | Min lead | Direction | Qualifies | Flag |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for x in sorted(by(rows2, signal=pa.POS_CONTROL), key=lambda x: (pa.TASK_ORDER.index(x["task"]), x["seed"])):
        flag = "NO ONSET" if x["onset"] is None else ("lead below minimum" if not x["qualifies"] else "")
        L.append(f"| {x['task']} | {x['seed']} | {x['t50']} | {f(x['onset'])} | {f(x['lead'])} | "
                 f"{lead_str(x) if x['lead'] is not None else '—'} | {f(x['min_lead'])} | "
                 f"{dirs(x['direction'])} | {f(x['qualifies'])} | {flag} |")
    L.append("")
    L.append("### 3.5 Spearman correlation, onset vs t50 (batch 2, descriptive only)\n")
    L.append("| Signal | Task | n | ρ |")
    L.append("|---|---|---|---|")
    for s in pa.SIGNALS + [pa.POS_CONTROL]:
        for t in pa.TASK_ORDER:
            if s == pa.POS_CONTROL and t not in pa.POS_CONTROL_TASKS:
                continue
            xs = [x for x in by(rows2, task=t, signal=s) if x["onset"] is not None]
            rho = pa.spearman([x["onset"] for x in xs], [x["t50"] for x in xs])
            L.append(f"| `{lab(s)}` | {t} | {len(xs)} | {'—' if rho is None else format(rho, '.3f')} |")
    L.append("")

    # 4 replication
    L.append("## 4. Replication table\n")
    L.append("Confirmed = precursor in batch 1 and in batch 2 (deviation 1).\n")
    L.append("| Signal | Task | B1 qual/n | B1 dir | B1 median norm. lead | B2 qual/n | B2 dir | "
             "B2 median norm. lead | Same direction | Confirmed |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in pa.SIGNALS:
        for t in pa.TASK_ORDER:
            r = rep[(t, s)]
            v1, v2 = r["v1"], r["v2"]
            L.append(f"| `{lab(s)}` | {t} | {v1['n_qual']}/{v1['n_seeds']} | {dirs(v1['direction'])} | "
                     f"{f(v1['median_norm_lead'])} | {v2['n_qual']}/{v2['n_seeds']} | "
                     f"{dirs(v2['direction'])} | {f(v2['median_norm_lead'])} | "
                     f"{f(r['same_dir']) if v1['direction'] and v2['direction'] else '—'} | "
                     f"**{f(r['confirmed'])}** |")
    L.append("")
    L.append("Pooled, descriptive only (no verdict): all seeds of both batches.\n")
    L.append("| Signal | Task | Qualifying / n | Directions of qualifying (+/−) | Median norm. lead |")
    L.append("|---|---|---|---|---|")
    for s in pa.SIGNALS:
        for t in pa.TASK_ORDER:
            xs = by(rows1, task=t, signal=s) + by(rows2, task=t, signal=s)
            q = [x for x in xs if x["qualifies"]]
            npos = sum(x["direction"] > 0 for x in q)
            med = float(np.median([x["norm_lead"] for x in q])) if q else None
            L.append(f"| `{lab(s)}` | {t} | {len(q)}/{len(xs)} | {npos}/{len(q) - npos} | {f(med)} |")
    L.append("")

    # 5 controls
    L.append("## 5. Control results\n")
    L.append("Valid controls only, anchored at the snapped batch-1 mean t50 with the same onset "
             "rule; min lead = max(2 × interval, 0.05 × anchor). A control onset counts if it "
             "exists and lead ≥ min lead (any direction).\n")
    L.append("### 5.1 Counting onsets per task, control type and signal\n")
    L.append("| Signal | Task | Type | Counting / valid seeds | Directions (seeds 101-103) | ≥ 2 counting |")
    L.append("|---|---|---|---|---|---|")
    for s in pa.SIGNALS:
        for t in pa.TASK_ORDER:
            for ct, p in spec[(t, s)]["per"].items():
                den = f"{p['k']}/{p['n_valid']}" + (f" (of {p['n_total']} runs)" if p["n_valid"] != p["n_total"] else "")
                L.append(f"| `{lab(s)}` | {t} | {ct} | {den} | {' '.join(dirs(d) for d in p['dirs'])} | "
                         f"{f(p['nonspec'])} |")
    L.append("")
    L.append("### 5.2 Per-run control table\n")
    L.append("| Task | Type | Seed | Signal | Anchor | Onset | Lead | Min lead | Direction | Counts |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for t in pa.TASK_ORDER:
        for ct in CTRL_TYPES_BY_TASK[t]:
            for s in pa.SIGNALS:
                for x in sorted(by(rowsc, task=t, ctype=ct, signal=s), key=lambda x: x["seed"]):
                    L.append(f"| {t} | {ct} | {x['seed']} | `{lab(s)}` | {x['t50']} | {f(x['onset'])} | "
                             f"{f(x['lead'])} | {f(x['min_lead'])} | {dirs(x['direction'])} | "
                             f"{f(x['qualifies'])} |")
    L.append("")

    # 6 specificity
    L.append("## 6. Specificity table\n")
    L.append("Non-specific = at least 2 of 3 counting control onsets in either control type of "
             "that task (deviation 6).\n")
    L.append("| Signal | Task | Confirmed | Non-specific | Final status |")
    L.append("|---|---|---|---|---|")
    for s in pa.SIGNALS:
        for t in pa.TASK_ORDER:
            sp = spec[(t, s)]
            which = [ct for ct, p in sp["per"].items() if p["nonspec"]]
            L.append(f"| `{lab(s)}` | {t} | {f(rep[(t, s)]['confirmed'])} | "
                     f"{f(sp['nonspec'])}{' (' + ', '.join(which) + ')' if which else ''} | {sp['status']} |")
    L.append("")

    # 7 cross-task
    L.append("## 7. Cross-task (confirmed and specific signals only)\n")
    L.append(f"Rule on batch 2: confirmed and specific in all three tasks, same batch-2 direction "
             f"in all three, max/min of the three batch-2 median normalized leads ≤ "
             f"{f(a.max_lead_ratio)}. Batch-1 values shown alongside.\n")
    L.append("| Signal | Confirmed & specific in | Eligible (all 3) | B2 directions | B2 lead ratio | "
             "B1 directions | B1 lead ratio | Shared precursor |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in cross:
        s = c["signal"]
        d2 = "/".join(dirs(ver2[(t, s)]["direction"]) for t in pa.TASK_ORDER)
        d1 = "/".join(dirs(ver1[(t, s)]["direction"]) for t in pa.TASK_ORDER)
        L.append(f"| `{lab(s)}` | {', '.join(c['ok_tasks']) or 'none'} | {f(c['eligible'])} | {d2} | "
                 f"{f(c['ratio2']) if c['eligible'] else '—'} | {d1} | "
                 f"{f(c['ratio1']) if c['eligible'] else '—'} | {f(c['shared'])} |")
    L.append("")
    L.append("Directions are listed induction/ICL/grokking (precursor direction; — if no qualifying "
             "seed, mixed if qualifying seeds disagree).\n")

    # 8 outcome
    L.append("## 8. Outcome under the decision rules\n")
    L.append("Rules as written in docs/hypothesis.md, evaluated literally:\n")
    L.append(f"- **Supports** (at least one shared precursor): "
             f"{'met: ' + ', '.join(f'`{lab(s)}`' for s in shared) if shared else 'not met (no shared precursor in section 7)'}.")
    for R, desc in readings.items():
        L.append(f"- Reading {R}: {desc}.")
        L.append(f"  - **Partial** (a signal qualifies in induction and ICL only, or in only two "
                 f"tasks): " + ("met: " + "; ".join(f"`{lab(s)}` ({', '.join(ts)})" for s, ts in partial[R])
                                if partial[R] else "not met") + ".")
        L.append(f"  - **Refutes**, clause 1 (no signal qualifies in ≥ 2 tasks): "
                 f"{'met' if refutes1[R] else 'not met'}.")
    L.append(f"- **Refutes**, clause 2 (every candidate is non-specific), reading candidates as "
             f"confirmed signal × task cells ({len(confirmed_cells)}): "
             + ("no confirmed cells, so the clause is vacuous" if not confirmed_cells else
                ("met" if all_nonspec else "not met")) + ".")
    L.append("- **Inconclusive** conditions:")
    L.append(f"  - “a signal fails by one seed”: batch 1: "
             + (", ".join(f"`{lab(s)}` in {t}" for t, s in short1) or "none")
             + "; batch 2: " + (", ".join(f"`{lab(s)}` in {t}" for t, s in short2) or "none") + ".")
    L.append(f"  - “a control reaches 50% accuracy”: controls whose metric reaches 0.5 at some "
             f"evaluation: " + (", ".join(f"{x['task']} {x['ctype']} seed {x['seed']} "
                                         f"(first {x['ever']}, max {x['max_metric']:.3f})" for x in ctrl_ever) or "none")
             + "; controls that stay ≥ 0.5 (invalid under deviation 4): "
             + (", ".join(f"{x['task']} {x['ctype']} seed {x['seed']}" for x in ctrl_stay) or "none") + ".")
    if sens:
        L.append(f"  - “the result flips across the sensitivity grid”: in logs/sensitivity_results.md "
                 f"(S1, batch 1 only), {sens['changed']} of {sens['total']} per-task verdicts change in "
                 f"at least one variant; shared candidate in any variant: {sens['shared_line']} "
                 "The S1 grid has not been run on batch 2 or on the controls.")
    else:
        L.append("  - “the result flips across the sensitivity grid”: logs/sensitivity_results.md not found.")
    L.append("")
    labels_by_reading = {R: applies[R] + (["Inconclusive"] if inconc else []) for R in readings}
    distinct = {tuple(v) for v in labels_by_reading.values()}
    single = len(distinct) == 1 and len(next(iter(distinct))) == 1
    L.append("**Which outcome applies:** " + (
        f"{next(iter(distinct))[0]} under every reading listed above."
        if single else
        "the rules do not give a single label for this case. Labels whose conditions are met: "
        + "; ".join(f"reading {R}: {', '.join(v) or 'none'}" for R, v in labels_by_reading.items())
        + ". Ambiguities: (i) whether “qualifies” in Partial/Refutes means confirmed-and-specific "
          "(A) or confirmed (B); (ii) the rules do not rank labels when more than one condition "
          "is met; (iii) “a control reaches 50% accuracy” can be read as reaching 0.5 at any "
          "evaluation (Controls section: “never reaches 0.5”) or as staying ≥ 0.5 (deviation 4); "
          "(iv) “fails by one seed” and “flips across the sensitivity grid” do not say whether "
          "they refer to batch 1, batch 2, any signal, or only signals relevant to a claim; "
          "(v) Inconclusive “triggers the extra-seed rule”, but S2 (seeds 6-10) is the extra-seed "
          "run itself and the pre-registration does not say what follows after it. "
          "No reading is chosen here.") + "\n")

    # 9 figures
    L.append("## 9. Figures\n")
    for t in pa.TASK_ORDER:
        L.append(f"- {pa.TASK_LABEL[t]}: [{rel(figs[t])}]({rel(figs[t])})")
    L.append("")
    L.append("Blue = batch 1, orange = batch 2, gray solid = C1 control, gray dashed = C2 control "
             "(valid controls only). x = (step − anchor)/anchor with anchor = the run's own t50 for "
             "jump runs and the snapped batch-1 mean t50 for controls; y = smoothed signal − the "
             "run's reference median m.\n")

    # 10 implementation choices
    L.append("## 10. Implementation choices\n")
    items = [
        "Batch membership comes from `seed` in config.json: 1-5 batch 1, 6-10 batch 2; controls "
        "are runs whose directory contains `_ctrl_`, with `experiment_name` mapped to task and "
        "type (C1: grokking_ctrl_wd0, icl_ctrl_fixedonly; C2: *_ctrl_shuffle) and seeds 101-103.",
        "All onset, lead, qualification, precursor and shared-candidate computations call the "
        "functions of precursor_analysis.py (`analyse`, `verdicts`, `cross_task`) with its "
        "pre-registered defaults; batch 1, batch 2 and controls are passed separately so each "
        "batch is analysed on its own.",
        "t50 = stays-above 0.5 on metrics.csv (`stays_above_step` from jump_detection.py). Batch-1 "
        "t50 is also checked against logs/run_results.md. For every jump run it is cross-checked "
        "against the same metric column in signals.csv; disagreements are listed in section 11 "
        "and metrics.csv is used.",
        "Controls: validity uses the stays-above rule on metrics.csv (deviation 4). For valid "
        "controls the run's “t50” is set to the anchor, so the reference window, onset window, "
        "direction and minimum lead all use the anchor. The anchor is the batch-1 mean t50 "
        "recomputed from the data (and checked equal to the pre-registered 296/816/7525), snapped "
        "to the nearest checkpoint step in the control's signals.csv.",
        "Non-specific: counting onsets ≥ 2 in either control type of the task. Induction has only "
        "a C2 control. If a control type had fewer than 3 valid seeds, the count is shown out of "
        "the valid seeds and the threshold of 2 is unchanged.",
        "Confirmed = precursor (pre-registered rule) in batch 1 and in batch 2. The rules do not "
        "require the direction to match between batches; the replication table shows whether it "
        "does.",
        "Section 7 applies the shared-precursor rule to batch-2 directions and batch-2 median "
        "normalized leads, as instructed; batch-1 values are shown but not used.",
        "Window-limited onsets (deviation 2) are marked WL; their normalized leads are included "
        "in medians unchanged (no decision rule is changed).",
        "The rise-time comparison (deviation 3) uses t90 − t50 from metrics.csv and is not used "
        "for any verdict.",
        "Section 8 lists every decision-rule label whose condition is met under each reading; it "
        "does not choose between them.",
        "No signals, parameters or windows other than the pre-registered ones were run.",
    ]
    L += [f"{i}. {s}" for i, s in enumerate(items, 1)]
    L.append("")

    # 11 unexpected
    L.append("## 11. Anything unexpected\n")
    un = []
    if notes["t50_check"]:
        un.append("t50 from metrics.csv differs from t50 recomputed on signals.csv: " + "; ".join(notes["t50_check"]))
    un += notes["anchor"]
    for name, nt in (("batch 2", notes2), ("controls", notesc)):
        if not nt:
            continue
        if nt["spacing"]:
            un.append(f"{name}: non-uniform checkpoint spacing: " + "; ".join(nt["spacing"]))
        if nt["nan"]:
            un.append(f"{name}: NaNs beyond step-0 cka_prev: " + "; ".join(nt["nan"]))
        if nt["t50_not_ckpt"]:
            un.append(f"{name}: t50/anchor not a checkpoint step: " + "; ".join(nt["t50_not_ckpt"]))
        if nt["floored"]:
            un.append(f"{name}: sigma at floor in {len(nt['floored'])} signal-runs: " + "; ".join(nt["floored"]))
        if nt["few_ref"]:
            un.append(f"{name}: reference window with < 3 checkpoints: " + "; ".join(nt["few_ref"]))
    wl2 = [x for x in rows2 if x["onset_at_first_candidate"]]
    if wl2:
        un.append(f"batch 2: {len(wl2)} window-limited onsets: " + "; ".join(
            f"{x['task']} seed {x['seed']} `{lab(x['signal'])}`" for x in wl2))
    wlc = [x for x in rowsc if x["onset_at_first_candidate"]]
    if wlc:
        un.append(f"controls: {len(wlc)} onsets at the first checkpoint after the reference window: "
                  + "; ".join(f"{x['task']} {x['ctype']} seed {x['seed']} `{lab(x['signal'])}`" for x in wlc))
    pc = by(rows2, signal=pa.POS_CONTROL)
    if any(not x["qualifies"] for x in pc):
        un.append("batch-2 positive control not qualifying: " + "; ".join(
            f"{x['task']} seed {x['seed']} (onset {f(x['onset'])})" for x in pc if not x["qualifies"]))
    dd = [(t, s) for (t, s), r in rep.items() if r["confirmed"] and not r["same_dir"]]
    if dd:
        un.append("confirmed in both batches but with different directions: "
                  + ", ".join(f"`{lab(s)}` in {t}" for t, s in dd))
    flips = [(t, s) for (t, s), r in rep.items() if r["v1"]["precursor"] != r["v2"]["precursor"]]
    if flips:
        un.append(f"{len(flips)} signal × task cells are a precursor in one batch but not the other: "
                  + ", ".join(f"`{lab(s)}` in {t} (B1 {f(rep[(t, s)]['v1']['precursor'])}, "
                              f"B2 {f(rep[(t, s)]['v2']['precursor'])})" for t, s in flips))
    mixed = [(t, s) for (t, s), v in ver2.items() if v["direction"] == "mixed"]
    if mixed:
        un.append("batch 2: qualifying seeds disagree in direction: "
                  + ", ".join(f"`{lab(s)}` in {t}" for t, s in mixed))
    invalid = [x for x in ctrl_items if not x["valid"]]
    if invalid:
        un.append("invalid controls: " + ", ".join(x["name"] for x in invalid))
    L += [f"- {u}" for u in un] if un else ["Nothing flagged by the automatic checks."]
    L.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""S1 sensitivity grid for the pre-registered precursor analysis (docs/hypothesis.md).

    python -m checkpoints_analysis.sensitivity          # writes logs/sensitivity_results.md
    python -m checkpoints_analysis.sensitivity --seeds 6,7,8,9,10 --out logs/sensitivity_batch2.md

Uses the analysis functions of checkpoints_analysis/precursor_analysis.py unchanged
(its defaults are the pre-registered values). Each variant changes ONE parameter
from the default; no other parameters are tried. --seeds selects the main runs
(default: batch 1 = seeds 1-5; 1-4 for grokking, whose seed 5 is _INCOMPLETE).
Writes only the report: no figures. For batch 1, logs/precursor_results.md is
read (section 9 list) but never written. For any other seed set, t50 is checked
against, and the window-limited list read from, logs/confirmation_results.md
(sections 3.1 and 11), which is never written.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from checkpoints_analysis import precursor_analysis as pa

REPO_ROOT = Path(__file__).resolve().parent.parent
BATCH1_SEEDS = {"induction": {1, 2, 3, 4, 5}, "icl": {1, 2, 3, 4, 5}, "grokking": {1, 2, 3, 4}}
DEFAULT_SEEDS = {1, 2, 3, 4, 5}
BATCH2_SEEDS = {6, 7, 8, 9, 10}
CONFIRMATION_REPORT = REPO_ROOT / "logs" / "confirmation_results.md"


def parse_seeds(s):
    return {int(x) for x in s.split(",") if x.strip()}


def batch_label(seeds):
    if seeds == DEFAULT_SEEDS:
        return "batch 1"
    if seeds == BATCH2_SEEDS:
        return "batch 2"
    return "seeds " + ",".join(map(str, sorted(seeds)))


def parse_confirmation_t50(path: Path):
    """{(task, seed): t50} from the per-run table (section 3.1) of confirmation_results.md."""
    sec = path.read_text(encoding="utf-8").split("### 3.1", 1)[1].split("### 3.2", 1)[0]
    out = {}
    for line in sec.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) > 3 and cells[0] in pa.TASK_ORDER and cells[2].isdigit():
            out[(cells[0], int(cells[2]))] = int(cells[3])
    return out

# (variant id, description, CLI args passed to precursor_analysis.parse_args)
VARIANTS = [
    ("default", "pre-registered defaults", []),
    ("smooth3", "smoothing window 3", ["--smooth-window", "3"]),
    ("smooth9", "smoothing window 9", ["--smooth-window", "9"]),
    ("band2", "band 2 sigma", ["--band-k", "2"]),
    ("band5", "band 5 sigma", ["--band-k", "5"]),
    ("ref5-15", "reference window 5-15% of t50", ["--ref-lo", "0.05", "--ref-hi", "0.15"]),
    ("ref10-40", "reference window 10-40% of t50", ["--ref-lo", "0.10", "--ref-hi", "0.40"]),
    ("t50fc", "first-crossing t50", ["--t50-mode", "first_crossing"]),
    ("minlead2", "minimum lead 2% of t50", ["--min-lead-frac", "0.02"]),
    ("minlead10", "minimum lead 10% of t50", ["--min-lead-frac", "0.10"]),
]
WINDOW_VARIANTS = ["default", "ref5-15", "ref10-40"]


def run_variant(cli, seeds=DEFAULT_SEEDS):
    a = pa.parse_args(["--seeds", ",".join(map(str, sorted(seeds)))] + cli)
    runs, _ = pa.load_runs(a)
    runs = [r for r in runs if r["seed"] in seeds]
    if seeds <= DEFAULT_SEEDS:
        mismatches, _ = pa.check_t50(runs, a)
        # check_t50 also flags run_results rows absent from `runs`; batch filtering
        # can only remove seeds outside the selected set, so ignore exactly those.
        mismatches = [m for m in mismatches if "not among the runs" not in m
                      or any(f"{t} seed {s} " in m for t in BATCH1_SEEDS for s in BATCH1_SEEDS[t]
                             if s in seeds)]
    else:
        # logs/run_results.md has batch-1 rows only. Stays-above t50 is checked against the
        # per-run table of logs/confirmation_results.md; first-crossing t50 has no reference
        # table for these seeds and is not checked.
        mismatches = []
        if a.t50_mode == "stays_above":
            ref = parse_confirmation_t50(CONFIRMATION_REPORT)
            for r in runs:
                if ref.get((r["task"], r["seed"])) != r["t50"]:
                    mismatches.append(f"{r['task']} seed {r['seed']}: computed t50={r['t50']}, "
                                      f"{CONFIRMATION_REPORT.name} t50={ref.get((r['task'], r['seed']))}")
        if any(r["t50"] is None for r in runs):
            mismatches.append("a run has no t50 under this t50 mode")
    if mismatches:
        print("STOP: t50 differs from logs/run_results.md; nothing written.", file=sys.stderr)
        for m in mismatches:
            print("  " + m, file=sys.stderr)
        sys.exit(2)
    rows, notes = pa.analyse(runs, a)
    ver = pa.verdicts(rows, runs, a)
    cross = pa.cross_task(ver, a)
    # first onset candidate per (task, seed, signal) for the drift diagnostic
    first_cand = {}
    for r in runs:
        for name, (steps, _sm, _res) in r["smoothed"].items():
            c = steps[(steps > a.ref_hi * r["t50"]) & (steps <= r["t50"])]
            first_cand[(r["task"], r["seed"], name)] = int(c[0]) if len(c) else None
    return {"a": a, "runs": runs, "rows": rows, "ver": ver, "cross": cross,
            "first_cand": first_cand}


def parse_window_limited(path: Path, batch1=True):
    """The window-limited list from section 9 of logs/precursor_results.md (batch 1) or the
    batch-2 line of section 11 of logs/confirmation_results.md (any other seed set)."""
    text = path.read_text(encoding="utf-8")
    if batch1:
        sec = text.split("## 9.", 1)[1]
        line = next(l for l in sec.splitlines() if l.startswith("- Onset at the first checkpoint"))
        items = line.split(": ", 1)[1].split("; ")
    else:
        sec = text.split("## 11.", 1)[1]
        line = next(l for l in sec.splitlines()
                    if l.startswith("- batch 2: ") and "window-limited onsets" in l)
        items = line.split(": ", 2)[2].split("; ")
    label_to_sig = {pa.sig_label(s): s for s in pa.SIGNALS + [pa.POS_CONTROL]}
    out = []
    for it in items:
        m = re.fullmatch(r"(\w+) seed (\d+) `(.+)`", it.strip())
        if not m:
            sys.exit(f"cannot parse section-9 item: {it!r}")
        out.append((m.group(1), int(m.group(2)), label_to_sig[m.group(3)]))
    n_claimed = int(re.search(r"in (\d+) signal-runs" if batch1 else r"batch 2: (\d+) window",
                              line).group(1))
    if n_claimed != len(out):
        sys.exit(f"section 9 says {n_claimed} signal-runs but lists {len(out)}")
    return out


def fmt(x, nd=3):
    return pa.f(x, nd)


def vcell(v):
    if v["precursor"]:
        return f"**yes** {pa.dirs(v['direction'])} ({v['n_qual']}/{v['n_seeds']}, {fmt(v['median_norm_lead'], 2)})"
    d = pa.dirs(v["direction"]) if v["direction"] == "mixed" else ""
    return f"no ({v['n_qual']}/{v['n_seeds']}{', mixed dir' if d else ''})"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "logs" / "sensitivity_results.md")
    ap.add_argument("--default-report", type=Path, default=None,
                    help="report holding the window-limited list (default: logs/precursor_results.md "
                         "for batch 1, logs/confirmation_results.md otherwise)")
    ap.add_argument("--seeds", type=parse_seeds, default=set(DEFAULT_SEEDS),
                    help="comma-separated main-run seeds (default: batch 1 = 1,2,3,4,5)")
    args = ap.parse_args(argv)
    is_b1 = args.seeds == DEFAULT_SEEDS
    if args.default_report is None:
        args.default_report = (REPO_ROOT / "logs" / "precursor_results.md" if is_b1
                               else CONFIRMATION_REPORT)
    if not is_b1 and args.out.resolve().name == "sensitivity_results.md":
        sys.exit("refusing to overwrite the batch-1 report with a different seed set; pass --out")
    blabel = batch_label(args.seeds)

    res = {vid: run_variant(cli, args.seeds) for vid, _, cli in VARIANTS}
    base = res["default"]
    keys = [(t, s) for s in pa.SIGNALS for t in pa.TASK_ORDER]

    def differs(v, b):
        return v["precursor"] != b["precursor"] or (
            v["precursor"] and b["precursor"] and v["direction"] != b["direction"])

    diffs = {vid: [k for k in keys if differs(res[vid]["ver"][k], base["ver"][k])]
             for vid, _, _ in VARIANTS[1:]}
    cand = {vid: [c["signal"] for c in res[vid]["cross"] if c["candidate"]] for vid, _, _ in VARIANTS}

    cmd = "python -m checkpoints_analysis.sensitivity" + (
        "" if is_b1 else f" --seeds {','.join(map(str, sorted(args.seeds)))} --out "
        + args.out.resolve().relative_to(REPO_ROOT).as_posix())
    L = [f"# S1 sensitivity grid ({blabel})\n",
         f"Generated by `{cmd}`. Do not edit by hand.\n"]

    # ---- summary
    n_flip_keys = sorted({k for d in diffs.values() for k in d}, key=keys.index)
    any_cand = sorted({s for v in cand.values() for s in v})
    L.append("## Summary\n")
    L.append(f"- {len(VARIANTS) - 1} one-at-a-time variants plus the default, on {blabel.replace(' ', '-')} runs "
             f"({', '.join(f'{t} {len([r for r in base['runs'] if r['task'] == t])}' for t in pa.TASK_ORDER)}).")
    L.append("- Shared candidate (specificity pending) in any variant: "
             + ("none. No variant produces a signal that is a precursor in all three tasks with "
                "one direction and lead ratio ≤ 3." if not any_cand else
                ", ".join(f"`{pa.sig_label(s)}` in {', '.join(v for v in cand if s in cand[v])}"
                          for s in any_cand) + "."))
    L.append(f"- Per-task verdicts that change in at least one variant: {len(n_flip_keys)} of "
             f"{len(keys)} (signal × task) cells.")
    stable = [k for k in keys if k not in n_flip_keys]
    L.append(f"- Verdicts identical across all variants: {len(stable)} of {len(keys)}.")
    L.append("")

    # ---- provenance
    L.append("## Provenance\n")
    L.append(f"- HEAD: `{pa.git('rev-parse', 'HEAD')}`")
    L.append(f"- Last commit touching docs/hypothesis.md: "
             f"`{pa.git('log', '-1', '--format=%H %s', '--', 'docs/hypothesis.md')}`")
    dirty = pa.git("status", "--porcelain", "--", "docs/hypothesis.md", "checkpoints_analysis")
    L.append(f"- Uncommitted changes in docs/hypothesis.md or checkpoints_analysis/: "
             f"{'none' if not dirty else '`' + dirty.replace(chr(10), '; ') + '`'}")
    L.append("- Runs: " + ", ".join(f"{r['task']} s{r['seed']}" for r in base["runs"]))
    L.append(f"- Default parameters: " + ", ".join(
        f"`{k}={getattr(base['a'], k)}`" for k in pa.PARAM_KEYS))
    L.append("")

    L.append("## Variants\n")
    L.append("| Id | Change from default | Arguments |")
    L.append("|---|---|---|")
    for vid, desc, cli in VARIANTS:
        L.append(f"| {vid} | {desc} | `{' '.join(cli) or '(none)'}` |")
    L.append("")
    L.append("All other parameters are at the pre-registered default. With `t50fc`, onset "
             "windows, leads and minimum leads are computed from the first-crossing t50 "
             + ("(checked against the `FC t50` column of logs/run_results.md).\n" if is_b1 else
                "(not checked against a reference table: logs/run_results.md has batch-1 rows only "
                "and logs/confirmation_results.md reports stays-above t50 only). Stays-above t50 "
                "is checked against the per-run table (section 3.1) of "
                "logs/confirmation_results.md.\n"))

    # ---- per-variant verdicts (compact matrix)
    L.append("## Per-task precursor verdicts by variant\n")
    L.append("Cell: **yes** + direction (qualifying/seeds, median normalized lead) or no "
             "(qualifying/seeds). ◆ = differs from default (precursor flag or precursor "
             "direction changed).\n")
    for t in pa.TASK_ORDER:
        L.append(f"### {t}\n")
        L.append("| Variant | " + " | ".join(f"`{pa.sig_label(s)}`" for s in pa.SIGNALS) + " |")
        L.append("|---|" + "---|" * len(pa.SIGNALS))
        for vid, _, _ in VARIANTS:
            cells = []
            for s in pa.SIGNALS:
                c = vcell(res[vid]["ver"][(t, s)])
                if vid != "default" and (t, s) in diffs[vid]:
                    c = "◆ " + c
                cells.append(c)
            L.append(f"| {vid} | " + " | ".join(cells) + " |")
        L.append("")

    # ---- cross-task
    L.append("## Cross-task candidates by variant\n")
    L.append("| Variant | Signals precursor in all 3 tasks | Same direction | Lead ratio | "
             "Shared candidate (specificity pending) | Precursor in exactly 2 tasks |")
    L.append("|---|---|---|---|---|---|")
    for vid, _, _ in VARIANTS:
        cr = res[vid]["cross"]
        all3 = [c for c in cr if c["in_all"]]
        two = [c for c in cr if c["n_tasks"] == 2]
        L.append(f"| {vid} | " + (", ".join(f"`{pa.sig_label(c['signal'])}`" for c in all3) or "none")
                 + " | " + (", ".join(pa.f(c["same_dir"]) for c in all3) or "—")
                 + " | " + (", ".join(fmt(c["ratio"], 2) for c in all3) or "—")
                 + " | " + (", ".join(f"`{pa.sig_label(s)}`" for s in cand[vid]) or "none")
                 + " | " + (", ".join(f"`{pa.sig_label(c['signal'])}` ({', '.join(c['tasks'])})"
                                      for c in two) or "none") + " |")
    L.append("")

    # ---- differences from default
    L.append("## Verdicts that differ from the default run\n")
    for vid, desc, _ in VARIANTS[1:]:
        d = diffs[vid]
        if not d:
            L.append(f"- **{vid}** ({desc}): no differences.")
            continue
        parts = []
        for t, s in d:
            b, v = base["ver"][(t, s)], res[vid]["ver"][(t, s)]
            parts.append(f"`{pa.sig_label(s)}` in {t}: {vcell(b)} → {vcell(v)}")
        L.append(f"- **{vid}** ({desc}): " + "; ".join(parts) + ".")
    L.append("")
    L.append("Number of variants (of 9) in which each verdict differs from default:\n")
    L.append("| Signal | " + " | ".join(pa.TASK_ORDER) + " |")
    L.append("|---|---|---|---|")
    for s in pa.SIGNALS:
        L.append(f"| `{pa.sig_label(s)}` | " + " | ".join(
            f"{sum((t, s) in diffs[v] for v in diffs)} (default: {'yes' if base['ver'][(t, s)]['precursor'] else 'no'})"
            for t in pa.TASK_ORDER) + " |")
    L.append("")

    # ---- drift diagnostic
    wl = parse_window_limited(args.default_report, batch1=is_b1)
    if not is_b1:
        # the list read from the confirmation report must equal the default variant's own list
        own = {(x["task"], x["seed"], x["signal"]) for x in base["rows"] if x["onset_at_first_candidate"]}
        if own != set(wl):
            sys.exit(f"window-limited list in {args.default_report.name} differs from the default "
                     f"variant: {sorted(own ^ set(wl))}")
    where = "section 9" if is_b1 else "section 11 (batch-2 line)"
    L.append("## Drift diagnostic: window-limited signal-runs\n")
    L.append(f"The {len(wl)} signal-runs listed in {where} of `{args.default_report.name}` had their "
             "default onset at the first checkpoint after the reference window. For each "
             "reference window, the table gives the onset and the first onset candidate "
             "(first checkpoint with step > ref_hi × t50). “Pinned” = onset equals the first "
             "candidate. “Follows edge earlier” = with the 5-15% window (edge moved earlier) the "
             "onset is again pinned at the new edge. “Follows edge later” = with the 10-40% "
             "window (edge moved later) the onset is again pinned at the new edge. “—” in "
             "onset = no onset in that variant.\n")
    L.append("| Task | Seed | Signal | t50 | " + " | ".join(
        f"{w}: edge → onset" for w in WINDOW_VARIANTS) + " | Follows edge earlier | Follows edge later |")
    L.append("|---|---|---|---|" + "---|" * (len(WINDOW_VARIANTS) + 2))
    patterns = {}
    for t, seed, s in wl:
        onsets, pinned, cells = [], [], []
        for w in WINDOW_VARIANTS:
            row = next(x for x in res[w]["rows"]
                       if x["task"] == t and x["seed"] == seed and x["signal"] == s)
            fc = res[w]["first_cand"][(t, seed, s)]
            on = row["onset"]
            onsets.append(on)
            pin = on is not None and on == fc
            pinned.append(pin if on is not None else None)
            cells.append(f"{fmt(fc)} → {fmt(on)}" + (" (pinned)" if pin else ""))
        if not pinned[0]:
            sys.exit(f"default run no longer reproduces window-limited onset for {t} {seed} {s}")
        def follow(i):
            return "no onset" if onsets[i] is None else ("yes" if pinned[i] else "no")
        early = follow(WINDOW_VARIANTS.index("ref5-15"))
        late = follow(WINDOW_VARIANTS.index("ref10-40"))
        patterns[(early, late)] = patterns.get((early, late), 0) + 1
        t50 = next(r["t50"] for r in base["runs"] if r["task"] == t and r["seed"] == seed)
        L.append(f"| {t} | {seed} | `{pa.sig_label(s)}` | {t50} | "
                 + " | ".join(cells) + f" | {early} | {late} |")
    L.append("")
    L.append("Counts by pattern (follows edge earlier / follows edge later):\n")
    L.append("| Earlier | Later | Signal-runs |")
    L.append("|---|---|---|")
    for (e, l_), n in sorted(patterns.items(), key=lambda kv: -kv[1]):
        L.append(f"| {e} | {l_} | {n} |")
    L.append("")

    # ---- notes
    L.append("## Notes\n")
    L.append("- One parameter changed per variant; no combinations and no parameters outside the "
             "S1 list were run.")
    L.append("- Minimum-lead variants change only the fraction term; the minimum lead is still "
             "max(2 × checkpoint interval, fraction × t50).")
    L.append("- The reference-window variants change both the band (m and σ are re-estimated in the "
             "new window) and the start of the onset search, so a change of onset can come from "
             "either.")
    L.append("- Specificity against controls is still pending for every variant.")
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(args.out.resolve())


if __name__ == "__main__":
    main()

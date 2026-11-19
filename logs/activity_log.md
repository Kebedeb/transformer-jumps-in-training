# Activity log: precursor / emergence project

## Oct 1, 2026
Set up the codebase for the precursor project. Fresh repo from the course
code, tokenizer and text pipeline removed. The model now returns per-head
attention weights and the residual stream, the loss can skip unsupervised
positions, and I fixed a seeding bug in the original. Built the three task
generators (induction, grokking, ICL), the checkpoint + signal pipeline, and
W&B tracking. Unit tests pass and a short induction run worked end to end.

## Oct 19, 2026
Cleaned up leftovers from the repo setup: removed an old course test,
fixed a misnamed file so it imports, and added a missing dependency. Tests
pass. Next: running each task in full to see if they show a clear jump.

## Oct 20, 2026

Ran the first full training run of all three tasks (seed 1234, W&B on, signal
analysis off on purpose so I'm only looking at behavior before I define what
counts as a jump). Models are small: about 110k parameters for induction, 106k
for ICL (2 layers, width 64) and 228k for grokking (1 layer, width 128). The
model is trained from scratch on synthetic data generated on the fly, so there
are no data files.

Induction. Training data is a random sequence of 16 distinct tokens shown
twice, and the model has to predict the second copy by looking back at what
followed the same token the first time. Sample:
`41 7 59 12 30 3 22 48 15 60 9 35 27 52 18 1 | 41 7 59 12 30 3 22 48 15 60 9 35 27 52 18` -> next token `1`.
[insert induction graph] Held-out accuracy goes from chance (~0.02) to 1.0
by step ~200 and stays there for the rest of the 10k steps. It's a real jump
but I can't see anything before it: I only evaluate every 100 steps and save
a checkpoint every 50, so the whole transition is covered by a few points.

Grokking. Training data is modular addition, (a + b) mod 113, with 30% of all
pairs used for training and the rest held out. Sample: `17 95 =` -> `112`.
[insert grokking graph] This one looks the way I wanted. Train accuracy is
1.0 by step ~300 while validation stays near zero, creeps up to ~0.2 by step
8000, then jumps between ~9000 and ~11500 and is done by ~12300. Two
problems. Train accuracy collapses several times between steps ~700 and 2000
(down to ~0.05 at one point), which looks like optimizer instability, and
there's a smaller dip in both train and val around step 12200 right before
things settle. 30k steps is way more than needed since everything is flat
after ~12.5k.

ICL. Training data is eight item/label pairs followed by a query item, and
the model has to give the query's label. Half the sequences use a fresh
random mapping (answerable only from the context) and half use one fixed
mapping. Sample: `5 35 | 19 32 | 5 35 | 2 38 | 27 35 | 11 39 | 19 32 | 30 33 | 19` -> `32`.
[insert ICL graph] The fixed-mapping accuracy hits ~0.95 within the first 100
steps (it can just be memorized). The fresh-mapping accuracy sits around 0.3
from step ~100 to ~700, then jumps to 1.0 by step ~1000, so there is a real
plateau and jump, just early. 10k steps was far more than needed.

Note on the plots: the grokking and ICL panels were on W&B's default x-axis, where
one unit is one evaluation (100 steps), so I converted to training steps above.
I'll set all panels to the `step` axis from now on.

What I take from this: the jumps exist in all three tasks, but induction and
ICL happen so early, relative to how often I measure, that I wouldn't be able
to see a precursor even if there is one. Grokking needs its early instability
fixed. Also worth noting for the hypothesis: the fresh-mapping ICL task is
basically the same lookup as induction, so grokking is the more informative
case for whether a precursor is really shared.

Next: rerun with finer measurement for induction and ICL (evaluate every 10
steps, checkpoint every 10, shorter runs), add optional LR warmup (and grad
clipping if needed) for grokking and rerun at 15k steps, then run a second
seed of each to check the jumps aren't one-seed flukes. Then write the
hypothesis and jump definition before looking at any signals.

## Nov 15, 2026

I added checkpoints_analysis/jump_detection.py, which reads only config.json, metrics.csv and the checkpoint file listing of each runs/*_main run and writes logs/run_results.md. All induction and ICL seeds (1-5) and grokking seeds 1-4 completed with the expected checkpoint counts, while grokking seed 5 is excluded because it has no metrics.csv, only 46 of 151 checkpoint files, and a zero-byte checkpoint at step 4500. Using "first step after which the metric stays at or above the threshold", mean t50/t90 across seeds is 296/326 for induction, 816/900 for ICL, and 7525/9775 for grokking, with grokking by far the most variable (t90 from 6600 to 14800). Two caveats came out of this: ICL fresh-map accuracy is already above 0.1 at step 10 (chance is 0.125), so the 10-90% width is meaningless for ICL, and the stays-above rule is sensitive to late dips, for example grokking seed 1 first reaches 0.9 at step 12600 but a single dip at step 14700 pushes its t90 to 14800.

## Nov 17, 2026

I extended checkpoints_analysis/jump_detection.py so that logs/run_results.md now includes a rise-time column (t90 − t50) for all three tasks, dropped t10 and width for ICL because fresh-map accuracy starts above 0.1, and added first-crossing versions of every jump column as a sensitivity check next to the primary stays-above ones. The first-crossing and stays-above values are identical for every induction and ICL seed and for grokking seeds 2 and 4, and they differ only for grokking seed 1 (t90 12600 vs 14800, rise 3300 vs 5500) and seed 3 (t50 8200 vs 8700). Mean rise time is 30 steps for induction, 84 for ICL and 2250 for grokking (1825 under first crossing). The script now also scans runs/*_main_INCOMPLETE so the excluded grokking seed 5 run stays listed with its reasons after the folder was renamed.

Wrote and committed the hypothesis and analysis plan before looking at any
internal signals: six candidate signals that exist in all three tasks, a fixed
onset rule, a minimum lead time, and what counts as support, partial support or
refutation. Also fixed the controls (no-jump runs for each task) and stress
tests in advance so I can't tune them to the results. Next: compute the signals
for the 14 runs and run the precursor analysis.

Ran the pre-registered precursor analysis once, with the default settings, on
the 14 main runs. No signal is a shared precursor. Representation change (1 -
CKA) precedes the jump in all three tasks in the same direction, but its
normalized lead differs a lot (0.06 ICL, 0.31 induction, 0.63 grokking), and
attention entropy falls in induction and ICL but rises in grokking. The positive
control (induction pattern score) passes in all 10 induction/ICL runs. Many
leads are capped by the reference window, so they may be drift rather than
precursors. The pre-registered extra-seed rule triggered. Next: control runs
(no-jump versions of each task), seeds 6-10 as a confirmation batch, and the
sensitivity checks.

## Nov 18, 2026
Finished the control runs (no-jump versions of each task) and the confirmation
seeds 6-10, and wrote the sensitivity checks to logs/sensitivity_results.md.
Computed signals for the new runs and fixed the control rules in the
hypothesis doc before looking at them. Next: confirmation and specificity
analysis, then the architecture and learning-rate stress tests.



## Report from Claude
2026-11-19: Ran the pre-registered S3/S4 stress analysis (new script
checkpoints_analysis/stress_analysis.py, report logs/stress_results.md, figures
logs/figures/stress_*.png) on the nine runs with seeds 201-203; all were complete and all
reached a jump. The precursor rule does not say how many of 3 seeds must qualify,
so both readings (2 of 3 and 3 of 3) are reported without choosing, and
specificity is not assessed because the controls exist only for the main
configs. Added a --seeds option to sensitivity.py; with the default seeds it
reproduces logs/sensitivity_results.md except the git-state provenance lines.
With seeds 6-10 it wrote logs/sensitivity_batch2.md, where 5 of 18 per-task
verdicts change in at least one variant and no variant gives a shared candidate.
Matplotlib does not import under the default Python 3.14 because
site-packages/PIL/Image.py has been overwritten by an unrelated script, so the
stress analysis was run with `py -3.13`.

## Nov 19, 2026
Confirmation and control analysis is done. No signal is a shared precursor
across all three tasks. Only attention entropy replicates and survives the
controls, in ICL and grokking, and it goes the opposite direction in grokking.
Most of the other signals turned out to be generic training drift. Also found
that grokking took longer in the second batch of seeds, so I'm checking why.
Running the architecture and learning-rate stress tests next.




# Hypothesis and analysis plan (pre-registration)

Written 2026-11-17, before any internal signal (signals.csv) was opened.
Seen so far: accuracy curves and the jump table in logs/run_results.md only.
Any change made after seeing signals goes under "Deviations" with a date.

## Question
Do induction-head formation, grokking and ICL emergence share a common
internal precursor: a measurable signal that changes before the accuracy
jump, in the same direction, in all three tasks?

## Data (frozen)
14 runs: induction seeds 1-5, ICL seeds 1-5, grokking seeds 1-4 (seed 5
excluded: disk filled at step ~4500). Settings are listed in run_results.md.

## Definitions
- Main metric: induction `heldout_acc`, ICL `fresh_map_acc`, grokking `val_acc`.
- Jump time t50: first step after which the main metric stays >= 0.5 until the
  end of the run. Sensitivity: first crossing of 0.5, and t90 (stays >= 0.9).
- Lead of a signal in a run = t50 - t_on (t_on defined below).
  Normalized lead = lead / t50.
- Minimum meaningful lead, per run: max(2 x checkpoint interval, 5% of that
  run's t50). Checkpoint intervals: 10 (induction, ICL), 100 (grokking).

## Candidate signals (fixed list, all defined for all three tasks)
L = number of layers; "final residual" = resid{L}.
1. Total weight norm (`weight_norm_total`)
2. Effective rank of the final residual (`resid{L}_eff_rank`)
3. Participation ratio of the final residual (`resid{L}_part_ratio`)
4. Representation change: 1 - `resid{L}_cka_prev`
5. Mean attention entropy over all heads
6. Maximum previous-token score over all heads

Positive control signal: induction pattern score (`attn_max_pattern`),
defined only for induction and ICL. It should not lag accuracy; if it does,
the detector or probe is suspect.
Exploratory (reported separately, never used for the main claim): per-head
values, per-block weight norms, grad norm, grokking spike events.

## Onset of a signal
Per signal and run: (1) smooth with a rolling median over 5 checkpoints;
(2) reference window = steps between 5% and 30% of t50; take its median m and
robust spread sigma = 1.4826 x MAD (with a small floor); (3) onset t_on =
earliest checkpoint after the reference window from which the smoothed signal
stays outside m +/- 3 sigma until t50; (4) if the signal is inside the band
just before t50, there is no onset. Direction = sign of (signal at t50 - m).

## Decision rules
- Precursor in a task: in all but one seed (4 of 5, or 3 of 4) the signal has
  an onset, lead >= minimum lead, and the direction agrees.
- Non-specific (discarded): the signal also gets an onset in >= 2 of 3 no-jump
  control runs of that task (controls anchored at the task's mean t50).
- Shared precursor: precursor and specific in all three tasks, same
  direction, and median normalized leads within a factor of 3.
- Supports: at least one shared precursor.
- Partial: a signal qualifies in induction and ICL only, or in only two tasks.
- Refutes (within this signal list and setup): no signal qualifies in >= 2
  tasks, or every candidate is non-specific.
- Inconclusive: a signal fails by one seed, a control reaches 50% accuracy
  (so it is not a no-jump run), or the result flips across the sensitivity
  grid. This triggers the extra-seed rule below.
Report all per-seed values. Also report, per task, the Spearman correlation
across seeds between onset time and t50 (descriptive only; n is small).

## Controls (defined now, run after the main analysis)
- C1, natural no-jump: grokking with weight_decay=0; ICL with icl_fraction=0.0.
- C2, shuffled targets: all three tasks, targets permuted across the batch each
  step (new option `training.shuffle_targets`).
- 3 seeds each (101-103), same steps and checkpoints as the main runs. A
  control counts only if its main metric never reaches 0.5.

## Stress tests (in priority order)
- S1: re-run the onset analysis over a grid: smoothing 3/5/9, band 2/3/5 sigma,
  reference windows 5-15% and 10-40%, first-crossing t50, minimum lead 2%/5%/10%.
  No new training.
- S2, extra-seed rule: if the outcome is inconclusive, or a signal qualifies in
  2 of 3 tasks, run seeds 6-10 on all three tasks before concluding. Fixed now
  so seeds can't be added selectively.
- S3, architecture match: grokking with 2 layers and width 64 (3 seeds), to
  check the cross-task comparison isn't an architecture artifact.
- S4, hyperparameters: induction at lr 1e-3 and ICL at lr 3e-4 (3 seeds each).

## Known limitations
4-5 seeds per task; small models; synthetic tasks; signals computed on fixed
probe inputs; grokking uses a different architecture (1 layer, width 128)
until S3; the fresh-map ICL task is solvable by an induction-style lookup, so
induction-ICL agreement is less surprising than agreement with grokking;
grokking shows training spikes and a staircase in validation accuracy that
make its jump time less sharp.

## Deviations and clarifications
- 2026-11-17 (after seeing batch-1 signals, before any seed 6-10 run was
  analysed):
  1. Seeds 1-5 (1-4 for grokking) are batch 1 and have been seen. Seeds 6-10 are
     the S2 confirmation batch, analysed on their own with the same rules (4 of 5
     seeds per task). A signal is confirmed only if it qualifies in both batches.
  2. Onsets pinned at the first checkpoint after the reference window are
     reported as "window-limited". Their normalized leads are lower bounds, not
     lead sizes. No decision rule is changed.
  3. Exploratory reading aid, not used for verdicts: compare each lead with the
     rise time (t90 - t50); a lead shorter than the rise overlaps the transition.
  4. A control run counts only if its main metric never stays at or above 0.5
     (same stays-above rule as t50). Controls that reach 0.5 are reported but
     not used for specificity.
  5. Controls are anchored at the batch-1 mean t50 (induction 296, ICL 816,
     grokking 7525), snapped to the nearest checkpoint step, and analysed with
     the same onset rule. A control onset counts if it exists and lead >=
     minimum lead (any direction; direction is reported).
  6. Specificity is evaluated per control type (C1, C2), 3 seeds each: a signal is
     non-specific for a task if it gets a counting onset in at least 2 of 3
     seeds of either control type.

  2026-11-19 (written AFTER seeing confirmation_results.md; post hoc):
  7. "Qualifies" in Partial/Refutes is reported under both readings (A: confirmed
     and specific; B: confirmed only). Both give Partial.
  8. When Partial and Inconclusive conditions are both literally met, the
     headline is Partial and the Inconclusive triggers are reported as
     limitations. The pre-registration did not rank the labels.
  9. Any analysis not in the pre-registration (e.g. direction-agnostic entropy
     change) is labelled exploratory and cannot change the outcome label.
  10. S3/S4 are robustness checks. They are reported under both readings of the
      n=3 precursor rule (A: at least 2 of 3; B: 3 of 3), and they do not
      change the outcome label. A claim that fails in a stress config is
      reported as a limitation of that claim.
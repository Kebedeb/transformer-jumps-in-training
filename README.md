# Precursor signals of emergent capabilities

**Research question.** Induction-head formation, grokking, and in-context
learning (ICL) each appear as a sudden jump in capability during training. Is
there a common, measurable internal signal that shows up *before* the jump in
all three?

**Approach.** Train a small transformer (`TinyTransformerLM`) on three
synthetic tasks, one per phenomenon. Save dense step checkpoints. At every
checkpoint, compute candidate signals (attention patterns, representation
geometry, weight norms) and test whether any of them leads the behavioural
jump, and whether the same one works across all three tasks.

Work done in DNU Lab (Dickinson College). The model architecture is adapted
from a course repo by Prof. MacCormick.

---

## Setup

```bash
pip install -r requirements.txt
wandb login            # optional, only needed for experiment tracking
python -m unittest discover -s tests -v     # sanity check, should pass
```

**Run everything from the repo root**, using `python -m ...` for scripts inside
packages.

---

## Running experiments

Each experiment trains one task, then analyzes every checkpoint.

```bash
python -m experiments.run_induction
python -m experiments.run_grokking
python -m experiments.run_icl
```

Common options (all three scripts):

| Option | Meaning |
|---|---|
| `-O key=value` | Override a config value; dotted keys, repeatable |
| `-c path.json` | Use a different config file |
| `--skip-analysis` | Train only, no checkpoint analysis |

Examples:

```bash
# quick smoke test
python -m experiments.run_induction -O training.train_steps=300 \
    -O checkpoints.dense_every=50 -O checkpoints.sparse_every=0

# different seed, no W&B
python -m experiments.run_induction -O seed=7 -O wandb.enabled=false

# several seeds in a row
for s in 1 2 3; do python -m experiments.run_induction -O seed=$s; done

# train only
python -m experiments.run_grokking --skip-analysis
```

### Other entry points

```bash
# train directly (no analysis step)
python train_transformer.py -c configs/icl_v1.json -O training.train_steps=5000

# analyze an existing run (default: most recent run under runs/)
python -m checkpoints_analysis.analyze runs/<run_dir>
python -m checkpoints_analysis.analyze --device cpu
```

---

## Tasks

Every task follows one convention: `y[b, t]` is the target for the prediction
made at position `t` (already shifted), and `y == -100` means that position is
not supervised.

| Task | Phenomenon | Input -> target | Eval sets |
|---|---|---|---|
| `induction` | induction heads | a random sequence of distinct tokens shown twice; predict the second copy from the first | `heldout` |
| `grokking` | grokking | `[a, b, =]` -> `(a+b) mod p`; only a fraction of pairs is used for training | `train`, `val` |
| `icl` | in-context learning | `[item, label, ..., query]` -> label of query; fresh per-sequence mapping mixed with one fixed global mapping | `fresh_map` (pure ICL), `fixed_map` (in-weights) |

Each task also supplies a fixed **probe batch** (inputs for analysis) and an
**attention target mask** (where a head implementing the mechanism should look).

---

## Configuration

Configs live in `configs/` (`induction_v1.json`, `grokking_v1.json`,
`icl_v1.json`). Every run saves the exact config (including overrides) to its
run directory.

| Section | Keys |
|---|---|
| top level | `experiment_name`, `seed`, `device` (`auto`/`cpu`/`cuda`) |
| `task` | `name` plus task parameters (e.g. `vocab_size`, `half_len`, `p`, `train_frac`, `n_items`, `n_labels`, `n_shots`, `icl_fraction`) |
| `model` | `n_layers`, `n_heads`, `n_embd`, `mlp_multiplier` (`n_embd` must be divisible by `n_heads`) |
| `training` | `batch_size`, `train_steps`, `eval_interval`, `learning_rate`, `weight_decay`, `beta1`, `beta2` |
| `checkpoints` | `dense_every`, `dense_until`, `sparse_every`, `save_optimizer` |
| `output` | `create_run_dir`, `save_checkpoints`, `save_local_csv` |
| `wandb` | `enabled`, `project`, `entity`, `mode` (`online`/`offline`/`disabled`), `tags` |

Checkpoints are saved at step 0, every `dense_every` steps up to `dense_until`,
then every `sparse_every` steps, plus the final step. Set an interval to `0` to
disable it. Checkpoint density limits how finely a precursor's lead time over
the jump can be resolved.

---

## Outputs

Each run creates `runs/<timestamp>-<experiment_name>/`:

| File | Contents |
|---|---|
| `config.json` | exact config used (analysis rebuilds the task from it) |
| `metrics.csv` | training-time behaviour (loss, accuracy, grad norm, weight norm) |
| `checkpoints/ckpt_step_XXXXXXX.pt` | weights, model kwargs, config, step |
| `signals.csv` | one row per checkpoint: behaviour plus all analysis signals |

With W&B enabled, the same data is also logged online: the training run
(`job_type=train`) and the analysis run (`job_type=analysis`) share a group
named after `experiment_name`, so seeds overlay in the dashboard.
`metrics.csv` and `signals.csv` are a local backup (turn off with
`-O output.save_local_csv=false`).

### Signals computed per checkpoint

| Column pattern | Meaning |
|---|---|
| `<set>_loss`, `<set>_acc` | behaviour on each eval set (supervised positions only) |
| `attn_L{l}H{h}_pattern` | mean attention on the task's target keys (the induction score for the induction task) |
| `attn_L{l}H{h}_prev_token` | mean attention from position q to q-1 |
| `attn_L{l}H{h}_entropy` | entropy of the head's attention distribution |
| `attn_max_pattern` | best pattern score over all heads |
| `resid{i}_eff_rank`, `resid{i}_part_ratio` | how many dimensions the residual stream uses |
| `resid{i}_mean_norm` | mean activation norm |
| `resid{i}_cka_prev` | similarity of representations to the previous checkpoint |
| `weight_norm_total`, `weight_norm_block{l}_{attn,mlp}` | weight norms |

Residual index 0 is the embedding; index `i+1` is the output of block `i`.

### Reading W&B data back for analysis

```python
from checkpoints_analysis.wandb_io import fetch_runs
runs = fetch_runs("precursor-emergence", "induction_v1", "analysis")
```

This uses `scan_history()`, which returns every logged row. Do not use
`run.history()` for jump-time analysis: it downsamples to about 500 points.

Offline runs: use `-O wandb.mode=offline`, then upload later with
`wandb sync <path to offline run folder>`.

---

## Repo layout

```
transformer_model.py       the model (per-head attention + residual stream on request)
train_transformer.py       trainer (batches from a Task, checkpoints, logging)
train_transformer_helpers.py  settings, overrides, evaluation, checkpoint saving
transformer_helper.py      device, checkpoint listing/loading
seed_utils.py              seeding (python, numpy, torch) + batch generator
config_utils.py            config loading, run directories
tracking.py                Weights & Biases wrapper (no-op if disabled)
tasks/                     induction, grokking, ICL data generators
checkpoints_analysis/      attention_stats, representation_stats, analyze, wandb_io
experiments/               run_induction / run_grokking / run_icl
configs/                   per-task JSON configs
tests/                     unit tests
logs/                      activity log
```

### Adding a new task

1. Create `tasks/<name>_task.py` with a subclass of `Task` (see `tasks/__init__.py`
   for the interface: `train_batch`, `eval_sets`, `probe_batch`, optional
   `attention_target_mask`).
2. Register it in `build_task` in `tasks/__init__.py`.
3. Add `configs/<name>_v1.json` and, optionally, `experiments/run_<name>.py`.

---

## Reproducibility notes

- Seeds control model initialisation, data sampling, and the task's fixed
  eval sets. Batch sampling uses its own generator, separate from the global RNG.
- Changing the seed changes the task's fixed sets (split, mapping, eval data)
  as well as initialisation.
- Task designs and hyperparameters are starting defaults, not tuned. Each task
  should show a clean, sharp jump before precursor analysis is trusted.

---

## Status

to be determined as results are obtained 
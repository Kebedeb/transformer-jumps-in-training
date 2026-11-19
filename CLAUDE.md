# Project context
Research question: do induction-head formation, grokking, and ICL emergence
share an internal precursor signal that appears before the behavioural jump?
Model: TinyTransformerLM on three synthetic tasks in tasks/. README.md has the
commands and the list of signals computed.

# Rules
- Run from the repo root with `python -m ...`.
- Don't edit configs/*_v1.json (locked). Use -O overrides for experiments and
  say so.
- Runs are in runs/<timestamp>-<experiment_name>/ (config.json, metrics.csv,
  checkpoints/, signals.csv). Analyse these local files, not W&B's
  downsampled history.
- Do NOT open or analyse signals.csv until docs/hypothesis.md is committed.
  metrics.csv (behaviour) is fine.
- Targets are pre-shifted by the task; y == -100 means unsupervised.
- Analysis must be scripts, not hand-typed numbers.
- After any notable task, add a short entry (3-5 sentences, plain prose, no
  bullets) to logs/activity_log.md.
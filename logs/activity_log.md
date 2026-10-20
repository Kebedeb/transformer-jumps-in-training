# Activity log: precursor / emergence project

## Oct 1, 2026
Set up the codebase for the precursor project. Fresh repo from the course
code, tokenizer and text pipeline removed. The model now returns per-head
attention weights and the residual stream, the loss can skip unsupervised
positions, and I fixed a seeding bug in the original. Built the three task
generators (induction, grokking, ICL), the checkpoint + signal pipeline, and
W&B tracking. Unit tests pass and a short induction run worked end to end.
Next: full runs to check each task shows a clean jump, then write down what
counts as a precursor before looking at any signals.


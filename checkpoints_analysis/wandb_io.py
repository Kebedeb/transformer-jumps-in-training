"""Pull complete (non-downsampled) run histories from W&B into pandas."""

from __future__ import annotations

import pandas as pd
import wandb


def fetch_runs(project: str, group: str, job_type: str, entity: str | None = None):
    """Return {run_name: (config, DataFrame sorted by step)} for one group.

    Uses scan_history(), which returns every logged row. Do NOT use
    run.history() here: it downsamples to ~500 points by default and would
    distort jump-time estimates.
    """
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"group": group, "jobType": job_type})
    out = {}
    for run in runs:
        df = pd.DataFrame(list(run.scan_history()))
        out[run.name] = (dict(run.config), df.sort_values("step").reset_index(drop=True))
    return out
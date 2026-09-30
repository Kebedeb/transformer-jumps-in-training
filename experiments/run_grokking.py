"""From repo root: python -m experiments.run_grokking [-O key=value]"""
from experiments.common import run_experiment

if __name__ == "__main__":
    run_experiment("grokking_v1.json")
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking import init_tracker


class TrackerTests(unittest.TestCase):
    def test_disabled_by_default_is_noop(self):
        t = init_tracker({}, name="x", job_type="train")
        self.assertFalse(t.enabled)
        t.log({"step": 1, "loss": 0.5})   # must not raise
        t.finish()

    def test_explicitly_disabled(self):
        t = init_tracker({"wandb": {"enabled": False}}, name="x", job_type="train")
        self.assertFalse(t.enabled)


if __name__ == "__main__":
    unittest.main()
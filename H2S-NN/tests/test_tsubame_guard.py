"""TSUBAME card T2: the 24 h guard of `src/utils/wandb_campaign.py: run_unit_sweep`.

A job killed during a trial (the 24 h limit) kills the parent with the worker, so the attempt has no bookkeeping and
`--resume` starts it again. The fake `subprocess.run` below raises `JobKilled` (a `BaseException`, as the job's
SIGKILL ends the parent) to stand for that; a returning fake is a worker that ran to its end.

(1) a worker that always dies is started exactly twice over three resumes; the line status is `exceeds_24h`;
(2) a worker that dies once, then finishes: the sweep completes and the open attempt is cleared;
(3) a state written before T2 (no counter) resumes as before.
Mutation (`docs/tsubame_evidence/t2_mutation.py`): limit = infinity -> (1) fails.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.utils import wandb_campaign as wc

SWEEP = {"name": "guard", "method": "bayes", "parameters": {"lr": {"values": [0.1]}}}


class JobKilled(BaseException):
    """The job ended by the scheduler: the parent dies with its worker."""


class FakeWandb:
    def sweep(self, **kwargs):
        return "guard-sweep"


def campaign(n_trials: int) -> wc.CampaignConfig:
    return wc.CampaignConfig(wandb_project="t", wandb_mode="offline", n_trials=n_trials, resume=True)


def unit() -> wc.CampaignUnit:
    return wc.CampaignUnit(run_index=0, num_runs=1, seed=0)


def sweep(tmp: Path, n_trials: int, worker) -> dict:
    """One launcher run (`--resume`) of one unit."""
    with mock.patch.dict(sys.modules, {"wandb": FakeWandb()}), \
            mock.patch("src.utils.wandb_campaign.subprocess.run", side_effect=worker):
        return wc.run_unit_sweep(
            {"model": "m"}, campaign=campaign(n_trials), unit=unit(), sweep_config=SWEEP, base_exp_name="cell",
            output_dir=tmp / "out", task_type="node_classification", base_config_fingerprint="", monitor="macro_f1",
            state_path=tmp / "state" / "tuning_state_run0.json", worker_entrypoint=Path(__file__),
            summarize=lambda records, monitor, path: {"n": len(records)})


def finishing_worker(calls: list):
    def worker(command, check):
        context = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        calls.append(context["attempt_index"])
        state_path = Path(context["state_path"])
        state = wc.load_tuning_state(state_path)
        state["records"] = [*state.get("records", []), {"trial_name": f"t{context['attempt_index']}",
                                                         "val_macro_f1": 0.5}]
        wc.save_tuning_state(state_path, state)
        wc.save_tuning_state(Path(context["status_path"]), {"ok": True, "worker_pid": 1})
        return SimpleNamespace(returncode=0)
    return worker


def dying_worker(calls: list):
    def worker(command, check):
        calls.append(json.loads(Path(command[-1]).read_text(encoding="utf-8"))["attempt_index"])
        raise JobKilled()
    return worker


def state_of(tmp: Path) -> dict:
    return json.loads((tmp / "state" / "tuning_state_run0.json").read_text(encoding="utf-8"))


class GuardTests(unittest.TestCase):
    def test_always_dying_trial_is_started_twice_then_exceeds_24h(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp, calls = Path(d), []
            for resume in range(3):
                with self.subTest(resume=resume):
                    if resume < 2:
                        with self.assertRaises(JobKilled):
                            sweep(tmp, 3, dying_worker(calls))
                    else:
                        with self.assertRaises(wc.TrialExceedsJobLimitError):
                            sweep(tmp, 3, dying_worker(calls))
            state = state_of(tmp)
        self.assertEqual(calls, [0, 0])  # attempt 0 started exactly twice, never a third time
        self.assertEqual(state["line_status"], "exceeds_24h")
        self.assertEqual(state["open_attempt"], {"attempt_index": 0, "starts": 2})
        self.assertEqual(state["attempted_trials"], 0)
        self.assertEqual(state["records"], [])

    def test_trial_dying_once_then_finishing_completes_the_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp, calls = Path(d), []
            with self.assertRaises(JobKilled):
                sweep(tmp, 2, dying_worker(calls))
            self.assertEqual(state_of(tmp)["open_attempt"], {"attempt_index": 0, "starts": 1})
            result = sweep(tmp, 2, finishing_worker(calls))
            state = state_of(tmp)
        self.assertEqual(calls, [0, 0, 1])
        self.assertEqual(result["n"], 2)
        self.assertEqual((state["attempted_trials"], state["successful_trials"]), (2, 2))
        self.assertNotIn("open_attempt", state)
        self.assertNotIn("line_status", state)

    def test_state_written_before_the_guard_resumes_as_before(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp, calls = Path(d), []
            # a pre-T2 state: one finished attempt of three, no `open_attempt` / `line_status`
            state = wc.initial_campaign_state(
                sweep_id="old-sweep", sweep_name="guard", campaign=campaign(3), unit=unit(), group="g",
                base_exp_name="cell", monitor="macro_f1", base_config_fingerprint="",
                sweep_config_fingerprint=wc.canonical_fingerprint(SWEEP))
            state.update({"attempted_trials": 1, "successful_trials": 1, "completed_trials": 1,
                          "records": [{"trial_name": "old", "val_macro_f1": 0.4}]})
            path = tmp / "state" / "tuning_state_run0.json"
            wc.save_tuning_state(path, state)
            result = sweep(tmp, 3, finishing_worker(calls))
            after = state_of(tmp)
        self.assertEqual(calls, [1, 2])  # resumes at attempt 1, as before
        self.assertEqual(result["n"], 3)
        self.assertEqual(after["sweep_id"], "old-sweep")
        self.assertEqual([r["trial_name"] for r in after["records"]], ["old", "t1", "t2"])
        self.assertEqual(after["attempted_trials"], 3)
        self.assertNotIn("open_attempt", after)
        self.assertNotIn("line_status", after)


if __name__ == "__main__":
    unittest.main()

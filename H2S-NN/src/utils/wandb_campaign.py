"""Model-agnostic W&B sweep orchestration.

This is the engine behind `scripts/run_h2snn_wandb_tuning.py` and
`scripts/run_baseline_tuning.py`: per-unit Bayesian+Hyperband sweeps, fingerprinted
resumable state, and one isolated worker process per trial.

Nothing here knows what a model is. Three things are injected by the caller:

* ``sweep_config`` -- the W&B sweep dict, built by whoever owns the search space.
* ``train_trial``  -- the per-trial fit, called inside the worker process.
* ``worker_entrypoint`` -- the CLI script to re-spawn for each trial. Every trial runs
  in a fresh interpreter so a CUDA fault or memory leak cannot poison the next one,
  which means the engine cannot use its own ``__file__`` -- that is a library module,
  not an entry point.

A "unit" is one tunable slot: a seed for full-graph tasks, or a (repetition, fold) pair
for repeated-CV graph classification.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

# Bumped whenever the on-disk state layout changes. Older states are rejected rather
# than migrated, so a stale remote sweep cannot reintroduce retired parameters.
TUNING_STATE_VERSION = 5


class TrainTrialFn(Protocol):
    """The per-trial fit contract, implemented by H2SNN and by each baseline adapter."""

    def __call__(
        self,
        raw_base: dict[str, Any],
        *,
        params: dict[str, Any],
        allowed_parameters: set[str] | frozenset[str] | None,
        run_index: int,
        num_runs: int,
        seed: int,
        base_exp_name: str,
        output_dir: Path,
        monitor: str,
        trial_name: str,
        unit_label: str,
        unit_name: str | None,
        record_extra: dict[str, Any] | None,
    ) -> dict[str, Any]: ...


@dataclass(slots=True)
class CampaignConfig:
    """W&B destination and trial budget for one campaign."""

    wandb_project: str = "h2snn"
    wandb_entity: str = ""
    wandb_mode: str = "online"
    wandb_group: str | None = None
    wandb_tags: tuple[str, ...] = ()
    n_trials: int = 20
    resume: bool = False
    sweep_id: str = ""

    @property
    def entity(self) -> str | None:
        return self.wandb_entity or None


@dataclass(slots=True)
class CampaignUnit:
    """One tunable slot: a seed, or a (repetition, fold) pair."""

    run_index: int
    num_runs: int
    seed: int
    label: str = "run"
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token(self) -> str:
        return self.name or f"{self.label}{self.run_index}"


def failure_details(exc: BaseException) -> dict[str, Any] | None:
    """The named condition of a failed trial (spec v1.16 K5): a stop condition (`tau`,
    `log_weight_range`, `non_finite_log_weight`, `non_finite_loss`) or `T_max`, with its value
    and epoch; None for any other error (still a failed trial, recorded by its message)."""
    from src.models.anchored_solve import PCGConvergenceError, StopConditionError

    if isinstance(exc, (StopConditionError, PCGConvergenceError)):
        return {"condition": getattr(exc, "condition", "T_max"),
                "value": getattr(exc, "value", None), "limit": getattr(exc, "limit", None),
                "epoch": getattr(exc, "epoch", None)}
    return None


def count_failures(failures: list[dict[str, Any]]) -> dict[str, int]:
    """Failed trials per condition (`other` when the trial failed for any other reason)."""
    counts: dict[str, int] = {}
    for attempt in failures:
        failure = attempt.get("failure") if isinstance(attempt, dict) else None
        key = str(failure.get("condition")) if isinstance(failure, dict) else "other"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def record_failed_attempt(state: dict[str, Any], attempt: dict[str, Any],
                          status: dict[str, Any] | None) -> None:
    """Append one failed trial to `state["failures"]`: no record, so no score reaches the
    Bayesian search; the attempt names the condition, epoch and value when one fired."""
    if isinstance(status, dict) and isinstance(status.get("failure"), dict):
        attempt["failure"] = status["failure"]
    entries = state.get("failures", [])
    if not isinstance(entries, list):
        entries = []
    entries.append(attempt)
    state["failures"] = entries
    state["failed_trials"] = len(entries)
    state["failed_by_condition"] = count_failures(entries)


def attach_failures(result: Any, failure_summary: dict[str, Any]) -> Any:
    """A unit's result (best record or cell summary) with its failed-trial counts, so the
    campaign summary can count failed trials per method and dataset."""
    if isinstance(result, dict):
        result["failed_trials"] = failure_summary["failed_trials"]
        result["failed_by_condition"] = failure_summary["failed_by_condition"]
    return result


def is_sweep_prune(exc: BaseException) -> bool:
    """Whether ``exc`` is Hyperband stopping this trial rather than the fit failing.

    ``wandb.agents.pyagent._terminate_thread`` stops an early-terminated trial by
    injecting a bare ``Exception`` into the run thread with ``PyThreadState_SetAsyncExc``.
    It is the ``Exception`` class itself and carries no arguments, which is what separates
    it from anything a model, a dataset, or CUDA can raise on its own.

    A pruned trial is a normal Hyperband outcome, not a broken worker: its partial result
    is discarded by design, and it must not be counted as a failure or the campaign will
    report every successful pruning as an error.
    """

    return type(exc) is Exception and not exc.args


def canonical_fingerprint(payload: Any) -> str:
    # `separators` is load-bearing: it must stay byte-identical or every previously
    # written tuning state stops resuming.
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_tuning_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Tuning state must be a JSON object: {path}")
    return payload


def save_tuning_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def completed_records_from_state(state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if state is None:
        return []
    records = state.get("records", [])
    if not isinstance(records, list):
        raise ValueError("Tuning state field 'records' must be a list.")
    return [record for record in records if isinstance(record, dict)]


def validate_resume_state(
    state: dict[str, Any] | None,
    *,
    state_path: Path,
    base_config_fingerprint: str,
    sweep_config_fingerprint: str,
) -> dict[str, Any]:
    if state is None:
        raise ValueError(
            f"--resume was requested but no tuning state exists at {state_path}. "
            "Start a fresh sweep with --resume omitted."
        )
    version = state.get("state_version")
    if version != TUNING_STATE_VERSION:
        raise ValueError(
            f"Refusing legacy tuning state at {state_path}: expected state_version="
            f"{TUNING_STATE_VERSION}, got {version!r}. Use a new experiment/state directory."
        )
    if state.get("base_config_fingerprint") != base_config_fingerprint:
        raise ValueError(
            f"Refusing to resume {state_path}: the resolved base configuration changed. "
            "Use the original configuration or start a new experiment/state directory."
        )
    if state.get("sweep_config_fingerprint") != sweep_config_fingerprint:
        raise ValueError(
            f"Refusing to resume {state_path}: the sweep search space changed. "
            "Start a new experiment/state directory so an old remote sweep cannot "
            "reintroduce unsafe parameters."
        )
    return state


def continuation_state_paths(state_path: Path) -> list[Path]:
    """Return numbered continuation states in ascending continuation order."""

    prefix = f"{state_path.stem}_continuation"
    candidates: list[tuple[int, Path]] = []
    for candidate in state_path.parent.glob(f"{prefix}*.json"):
        suffix = candidate.stem.removeprefix(prefix)
        if suffix.isdigit():
            candidates.append((int(suffix), candidate))
    return [candidate for _, candidate in sorted(candidates)]


def next_continuation_state_path(state_path: Path) -> Path:
    paths = continuation_state_paths(state_path)
    next_index = 1
    if paths:
        latest = paths[-1]
        prefix = f"{state_path.stem}_continuation"
        next_index = int(latest.stem.removeprefix(prefix)) + 1
    return state_path.with_name(
        f"{state_path.stem}_continuation{next_index}{state_path.suffix}"
    )


def resolve_campaign_state(
    state_path: Path,
    *,
    base_config_fingerprint: str,
    sweep_config_fingerprint: str,
) -> tuple[Path, dict[str, Any] | None, str]:
    """Select a resumable state or a fresh continuation state for one campaign unit.

    A compatible state resumes its original sweep. Missing units start a fresh sweep.
    Existing incompatible states are preserved, and a numbered continuation state is
    selected so the old sweep and its provenance remain available for inspection.
    """

    candidates = [state_path, *continuation_state_paths(state_path)]
    for candidate in candidates:
        state = load_tuning_state(candidate)
        if state is None:
            continue
        try:
            validated = validate_resume_state(
                state,
                state_path=candidate,
                base_config_fingerprint=base_config_fingerprint,
                sweep_config_fingerprint=sweep_config_fingerprint,
            )
        except ValueError:
            continue
        return candidate, validated, "resume"

    if not any(candidate.exists() for candidate in candidates):
        return state_path, None, "new"
    return next_continuation_state_path(state_path), None, "continuation"


def initial_campaign_state(
    *,
    sweep_id: str,
    sweep_name: str,
    campaign: CampaignConfig,
    unit: CampaignUnit,
    group: str,
    base_exp_name: str,
    monitor: str,
    base_config_fingerprint: str,
    sweep_config_fingerprint: str,
) -> dict[str, Any]:
    return {
        "state_version": TUNING_STATE_VERSION,
        "base_config_fingerprint": base_config_fingerprint,
        "sweep_config_fingerprint": sweep_config_fingerprint,
        "sweep_id": sweep_id,
        "sweep_name": sweep_name,
        "wandb_project": campaign.wandb_project,
        "wandb_entity": campaign.wandb_entity,
        "wandb_group": group,
        "base_exp_name": base_exp_name,
        "run_index": unit.run_index,
        "unit_label": unit.label,
        "unit_name": unit.token,
        "unit_metadata": dict(unit.metadata),
        "num_runs": unit.num_runs,
        "seed": unit.seed,
        "target_trials": campaign.n_trials,
        "attempted_trials": 0,
        "successful_trials": 0,
        "failed_trials": 0,
        "completed_trials": 0,
        "monitor": monitor,
        "records": [],
        "failures": [],
    }


def write_worker_status(path: Path, payload: dict[str, Any]) -> None:
    save_tuning_state(path, payload)


def run_campaign_worker(
    context_path: Path,
    *,
    train_trial: TrainTrialFn,
    known_parameters: set[str] | frozenset[str],
    validate_params: Callable[[dict[str, Any], set[str] | frozenset[str]], None] | None = None,
    validate_raw_base: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Run exactly one sweep trial inside this (fresh) worker process.

    The parent counts successes by re-reading the state file, so this must record
    exactly one record or raise.

    ``validate_raw_base`` re-parses the serialised base config before any training
    starts, so a context file that no longer resolves fails loudly here rather than
    part-way through a trial.
    """

    from src.utils.run_reports import metric_score

    with context_path.open("r", encoding="utf-8") as handle:
        context = json.load(handle)
    if not isinstance(context, dict):
        raise ValueError(f"Worker context must be a JSON object: {context_path}")

    state_path = Path(context["state_path"])
    status_path = Path(context["status_path"])
    state = validate_resume_state(
        load_tuning_state(state_path),
        state_path=state_path,
        base_config_fingerprint=str(context["base_config_fingerprint"]),
        sweep_config_fingerprint=str(context["sweep_config_fingerprint"]),
    )
    initial_successes = len(completed_records_from_state(state))

    try:
        import wandb  # type: ignore[import-not-found]

        raw_base = context["raw_base"]
        if validate_raw_base is not None:
            validate_raw_base(raw_base)
        allowed_parameters = set(
            str(key) for key in context.get("allowed_parameters", known_parameters)
        )
        entity = str(context.get("wandb_entity", "")) or None
        tags = list(context.get("wandb_tags", []))
        objective_error: str | None = None
        objective_failure: dict[str, Any] | None = None
        objective_pruned = False

        def objective() -> None:
            nonlocal objective_error, objective_pruned, objective_failure
            try:
                init_kwargs: dict[str, Any] = {
                    "project": context["wandb_project"],
                    "mode": context["wandb_mode"],
                    "group": context["wandb_group"],
                    "tags": tags,
                }
                if entity:
                    init_kwargs["entity"] = entity
                with wandb.init(**init_kwargs) as run:
                    params = dict(run.config)
                    if validate_params is not None:
                        validate_params(params, allowed_parameters)
                    trial_name = run.name or run.id
                    record = train_trial(
                        raw_base,
                        params=params,
                        allowed_parameters=allowed_parameters,
                        run_index=int(context["run_index"]),
                        num_runs=int(context["num_runs"]),
                        seed=int(context["seed"]),
                        base_exp_name=str(context["base_exp_name"]),
                        output_dir=Path(context["output_dir"]),
                        monitor=str(context["monitor"]),
                        trial_name=trial_name,
                        unit_label=str(context.get("unit_label", "run")),
                        unit_name=str(context.get("unit_name", "")) or None,
                        record_extra=(
                            dict(context["unit_metadata"])
                            if isinstance(context.get("unit_metadata"), dict)
                            else None
                        ),
                    )
                    record["wandb_run_id"] = run.id
                    record["wandb_run_name"] = run.name
                    record["wandb_sweep_id"] = context["sweep_id"]

                    latest_state = validate_resume_state(
                        load_tuning_state(state_path),
                        state_path=state_path,
                        base_config_fingerprint=str(context["base_config_fingerprint"]),
                        sweep_config_fingerprint=str(context["sweep_config_fingerprint"]),
                    )
                    records = completed_records_from_state(latest_state)
                    records.append(record)
                    latest_state["records"] = records
                    latest_state["successful_trials"] = len(records)
                    latest_state["completed_trials"] = len(records)
                    save_tuning_state(state_path, latest_state)

                    for key, value in record.items():
                        if isinstance(value, (int, float, str, bool)):
                            run.summary[f"best/{key}"] = value
                    wandb.log(
                        {
                            f"best/{str(context['monitor']).removeprefix('val_')}": metric_score(
                                record, str(context["monitor"])
                            )
                        }
                    )
            except Exception as exc:
                # A failed trial (spec v1.16 K5): nothing is logged under the sweep metric, so
                # no score reaches the Bayesian search; the named condition goes to the status.
                objective_failure = failure_details(exc)
                objective_pruned = is_sweep_prune(exc)
                objective_error = (
                    "early-terminated by the sweep's Hyperband bracket"
                    if objective_pruned
                    else f"{type(exc).__name__}: {exc}"[:1000]
                )
                raise

        wandb.agent(
            str(context["sweep_id"]),
            function=objective,
            count=1,
            project=str(context["wandb_project"]),
            entity=entity,
        )
        final_state = load_tuning_state(state_path)
        final_successes = len(completed_records_from_state(final_state))
        if final_successes != initial_successes + 1:
            if objective_pruned:
                # Hyperband killed the run thread part-way through the fit, so there is
                # no record to write. Exit cleanly and let the parent count it as a
                # pruned attempt rather than a failed one.
                write_worker_status(
                    status_path,
                    {
                        "ok": True,
                        "pruned": True,
                        "attempt_index": int(context["attempt_index"]),
                        "worker_pid": os.getpid(),
                    },
                )
                return
            detail = f" Last objective error: {objective_error}" if objective_error else ""
            failure = RuntimeError(
                "W&B worker returned without recording exactly one successful trial."
                f"{detail}"
            )
            failure.trial_failure = objective_failure  # type: ignore[attr-defined]
            raise failure
        write_worker_status(
            status_path,
            {
                "ok": True,
                "pruned": False,
                "attempt_index": int(context["attempt_index"]),
                "worker_pid": os.getpid(),
            },
        )
    except Exception as exc:
        write_worker_status(
            status_path,
            {
                "ok": False,
                "attempt_index": int(context["attempt_index"]),
                "worker_pid": os.getpid(),
                "error": f"{type(exc).__name__}: {exc}"[:1000],
                "failure": getattr(exc, "trial_failure", None) or failure_details(exc),
            },
        )
        raise


def run_unit_sweep(
    raw_base: dict[str, Any],
    *,
    campaign: CampaignConfig,
    unit: CampaignUnit,
    sweep_config: dict[str, Any],
    base_exp_name: str,
    output_dir: Path,
    task_type: str,
    base_config_fingerprint: str,
    monitor: str,
    state_path: Path,
    worker_entrypoint: Path,
    summarize: Callable[[list[dict[str, Any]], str, Path], dict[str, Any]],
    run_count: int = 1,
) -> dict[str, Any]:
    """Drive one unit's sweep to ``campaign.n_trials`` attempts and return its best record.

    Each attempt runs in a fresh interpreter spawned from ``worker_entrypoint``; success
    is counted by re-reading the fingerprinted state file rather than by trusting the
    child's exit code.
    """

    if campaign.sweep_id and not campaign.resume:
        raise ValueError(
            "--sweep_id is accepted only together with --resume and an existing "
            "fingerprinted local tuning state."
        )
    if campaign.sweep_id and run_count != 1:
        raise ValueError("--sweep_id is only unambiguous when --run_count 1 is used.")

    sweep_name = str(sweep_config["name"])
    unit_token = unit.token
    group = campaign.wandb_group or f"{base_exp_name}_tuning_{unit_token}"
    entity = campaign.entity
    sweep_config_fingerprint = canonical_fingerprint(sweep_config)

    primary_state_path = state_path
    existing_state = load_tuning_state(state_path)
    state: dict[str, Any] | None = None
    resume_action = "fresh"
    if campaign.resume:
        if campaign.sweep_id:
            # An explicit sweep ID remains an exact-resume request.
            state = validate_resume_state(
                existing_state,
                state_path=state_path,
                base_config_fingerprint=base_config_fingerprint,
                sweep_config_fingerprint=sweep_config_fingerprint,
            )
            resume_action = "resume"
        else:
            state_path, state, resume_action = resolve_campaign_state(
                state_path,
                base_config_fingerprint=base_config_fingerprint,
                sweep_config_fingerprint=sweep_config_fingerprint,
            )
            existing_state = state
        if state is not None:
            if campaign.sweep_id and str(state.get("sweep_id", "")) != campaign.sweep_id:
                raise ValueError(
                    f"--sweep_id={campaign.sweep_id!r} does not match the fingerprinted state "
                    f"sweep id {state.get('sweep_id')!r} in {state_path}."
                )
            sweep_id = str(state.get("sweep_id", ""))
            if not sweep_id:
                raise ValueError(f"Fingerprint-valid state has no sweep_id: {state_path}")
    elif existing_state is not None:
        raise ValueError(
            f"Refusing to overwrite existing tuning state at {state_path}. "
            "Use --resume for the same fingerprinted sweep or choose a new "
            "experiment/state directory."
        )

    if state is None:
        if campaign.sweep_id:
            raise ValueError(
                "An explicit --sweep_id requires a compatible existing local state."
            )
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - environment specific
            raise RuntimeError("W&B tuning requires the 'wandb' package.") from exc
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=campaign.wandb_project,
            entity=entity,
        )
        state = initial_campaign_state(
            sweep_id=str(sweep_id),
            sweep_name=sweep_name,
            campaign=campaign,
            unit=unit,
            group=group,
            base_exp_name=base_exp_name,
            monitor=monitor,
            base_config_fingerprint=base_config_fingerprint,
            sweep_config_fingerprint=sweep_config_fingerprint,
        )
        if campaign.resume:
            state["resume_action"] = resume_action
            if resume_action == "continuation":
                state["continuation_of"] = str(primary_state_path)
        save_tuning_state(state_path, state)

    attempted_trials = int(state.get("attempted_trials", 0))
    state["target_trials"] = campaign.n_trials
    save_tuning_state(state_path, state)
    remaining_attempts = max(0, campaign.n_trials - attempted_trials)

    for attempt_offset in range(remaining_attempts):
        attempt_index = attempted_trials + attempt_offset
        status_path = state_path.with_name(
            f"{state_path.stem}_worker_status_{attempt_index}.json"
        )
        context_path = state_path.with_name(f"{state_path.stem}_worker_context.json")
        worker_context = {
            "raw_base": raw_base,
            "run_index": unit.run_index,
            "unit_label": unit.label,
            "unit_name": unit_token,
            "unit_metadata": dict(unit.metadata),
            "num_runs": unit.num_runs,
            "seed": unit.seed,
            "base_exp_name": base_exp_name,
            "output_dir": str(output_dir),
            "monitor": monitor,
            "state_path": str(state_path),
            "status_path": str(status_path),
            "attempt_index": attempt_index,
            "sweep_id": str(sweep_id),
            "wandb_project": campaign.wandb_project,
            "wandb_entity": campaign.wandb_entity,
            "wandb_mode": campaign.wandb_mode,
            "wandb_group": group,
            "wandb_tags": list(
                dict.fromkeys((*campaign.wandb_tags, task_type, "tuning"))
            ),
            "allowed_parameters": sorted(sweep_config["parameters"]),
            "base_config_fingerprint": base_config_fingerprint,
            "sweep_config_fingerprint": sweep_config_fingerprint,
        }
        save_tuning_state(context_path, worker_context)

        before_state = load_tuning_state(state_path)
        before_successes = len(completed_records_from_state(before_state))
        result = subprocess.run(
            [
                sys.executable,
                str(Path(worker_entrypoint).resolve()),
                "--worker_context",
                str(context_path),
            ],
            check=False,
        )
        after_state = validate_resume_state(
            load_tuning_state(state_path),
            state_path=state_path,
            base_config_fingerprint=base_config_fingerprint,
            sweep_config_fingerprint=sweep_config_fingerprint,
        )
        after_successes = len(completed_records_from_state(after_state))
        after_state["attempted_trials"] = attempt_index + 1
        after_state["successful_trials"] = after_successes
        after_state["completed_trials"] = after_successes

        if after_successes != before_successes + 1:
            status = load_tuning_state(status_path)
            attempt = {
                "attempt_index": attempt_index,
                "return_code": int(result.returncode),
                "worker_pid": (
                    status.get("worker_pid") if isinstance(status, dict) else None
                ),
            }
            # A Hyperband prune reaches here exactly like a crash does -- no record was
            # written -- so the worker's status file is what tells the two apart.
            pruned = bool(isinstance(status, dict) and status.get("pruned"))
            if pruned:
                entries = after_state.get("pruned", [])
                if not isinstance(entries, list):
                    entries = []
                entries.append(attempt)
                after_state["pruned"] = entries
            else:
                attempt["error"] = (
                    str(status.get("error"))
                    if isinstance(status, dict) and status.get("error")
                    else f"worker exited with return code {result.returncode} "
                    "before recording a trial"
                )
                record_failed_attempt(after_state, attempt, status)
        after_state["failed_trials"] = len(after_state.get("failures", []))
        after_state["failed_by_condition"] = count_failures(after_state.get("failures", []))
        after_state["pruned_trials"] = len(after_state.get("pruned", []))
        save_tuning_state(state_path, after_state)

    final_state = validate_resume_state(
        load_tuning_state(state_path),
        state_path=state_path,
        base_config_fingerprint=base_config_fingerprint,
        sweep_config_fingerprint=sweep_config_fingerprint,
    )
    trial_records = completed_records_from_state(final_state)
    seed_dir = output_dir / base_exp_name
    seed_dir.mkdir(parents=True, exist_ok=True)
    with (seed_dir / f"tuning_trials_{unit_token}.json").open("w", encoding="utf-8") as handle:
        json.dump(trial_records, handle, indent=2)
    if not trial_records:
        raise RuntimeError(
            f"No W&B trials succeeded for run {unit.run_index}; "
            f"{final_state.get('failed_trials', 0)} isolated worker attempts failed. "
            f"Inspect {state_path} and the worker status files."
        )
    print(
        f"W&B run {unit.run_index}: attempted={final_state.get('attempted_trials', 0)} "
        f"succeeded={len(trial_records)} pruned={final_state.get('pruned_trials', 0)} "
        f"failed={final_state.get('failed_trials', 0)}",
        flush=True,
    )
    failures = final_state.get("failures", []) if isinstance(final_state, dict) else []
    failure_summary = {"failed_trials": len(failures),
                       "failed_by_condition": count_failures(failures), "failures": failures}
    with (seed_dir / f"tuning_failures_{unit_token}.json").open("w", encoding="utf-8") as handle:
        json.dump(failure_summary, handle, indent=2)
    return attach_failures(summarize(
        trial_records,
        monitor,
        seed_dir / f"best_tuned_{unit_token}.json",
    ), failure_summary)

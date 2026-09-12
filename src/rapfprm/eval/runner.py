"""Evaluation runner: score ProcessBench under one condition and write a run directory.

A run directory is self-describing — config, environment, predictions, per-step traces and
metrics all live together, so a number in the report can always be traced back to the code
and settings that produced it.

    runs/<name>/
      config.resolved.yaml   every setting, after defaults and overrides
      summary.json           environment, retrieval stats, headline metrics
      predictions.jsonl      one row per solution: gold vs predicted first-error index
      traces.jsonl           one row per scored step (only when run.save_traces)
      metrics.json           per-subset and average F1
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..data.processbench import Solution, group_by_subset, load_processbench
from ..prm.base import StepVerdict, build_backend
from ..retrieval.retriever import Reference, Retriever, load_retriever
from .metrics import score_all

logger = logging.getLogger(__name__)


@dataclass
class SolutionPrediction:
    uid: str
    subset: str
    gold_label: int
    predicted_label: int
    n_steps: int
    n_steps_scored: int
    #: Error type the model assigned at the step it flagged; None when it flagged nothing.
    predicted_error_type: str | None
    n_references_used: int

    @property
    def correct(self) -> bool:
        return self.gold_label == self.predicted_label

    def to_dict(self) -> dict:
        payload = self.__dict__.copy()
        payload["correct"] = self.correct
        return payload


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unversioned"


def _environment(cfg: Config) -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_revision": _git_revision(),
        "backend": cfg.prm.backend,
        "model_name": cfg.prm.model_name if cfg.prm.backend != "mock" else "mock",
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    return info


def score_solution(
    solution: Solution,
    backend,
    retriever: Retriever | None,
    cfg: Config,
) -> tuple[SolutionPrediction, list[dict]]:
    """Walk the solution step by step and return its first-error prediction."""
    predicted_label = -1
    predicted_error_type: str | None = None
    references_used = 0
    traces: list[dict] = []

    for index, step in enumerate(solution.steps):
        prev_steps = solution.steps[:index]

        references: list[Reference] = []
        if retriever is not None:
            references = retriever.retrieve(solution.problem, step)
        references_used += len(references)

        verdict: StepVerdict = backend.score_step(solution.problem, prev_steps, step, references)
        is_error = verdict.is_error(cfg.prm.correctness_threshold)

        if cfg.run.save_traces:
            traces.append(
                {
                    "uid": solution.uid,
                    "subset": solution.subset,
                    "step_index": index,
                    "gold_label": solution.label,
                    "is_error": is_error,
                    **verdict.to_dict(),
                    "reference_qids": [r.item.qid for r in references],
                    "reference_similarities": [
                        round(r.step_similarity, 4) for r in references
                    ],
                }
            )

        if is_error and predicted_label == -1:
            predicted_label = index
            predicted_error_type = verdict.error_type
            if cfg.prm.early_stop:
                break

    return (
        SolutionPrediction(
            uid=solution.uid,
            subset=solution.subset,
            gold_label=solution.label,
            predicted_label=predicted_label,
            n_steps=len(solution.steps),
            n_steps_scored=len(traces) if cfg.run.save_traces else 0,
            predicted_error_type=predicted_error_type,
            n_references_used=references_used,
        ),
        traces,
    )


def run_evaluation(cfg: Config) -> dict:
    """Execute one condition end to end and return its metrics."""
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.resolved.yaml")

    solutions = load_processbench(cfg.data)
    logger.info(
        "Loaded %d ProcessBench solutions across %s", len(solutions), list(cfg.data.subsets)
    )

    retriever = None
    if cfg.retrieval.enabled:
        retriever = load_retriever(cfg.retrieval, cfg.data.pool_dir)
        logger.info(
            "Retrieval ON — %d pool items, encoder %s", len(retriever.items), retriever.index.encoder_name
        )
    else:
        logger.info("Retrieval OFF — Condition A (baseline)")

    backend = build_backend(cfg.prm, cfg.prompt)
    logger.info("Backend: %s", backend.name)
    if cfg.prm.backend == "mock":
        logger.warning("MOCK BACKEND — these numbers are not results.")

    predictions_path = run_dir / "predictions.jsonl"
    traces_path = run_dir / "traces.jsonl"

    # Resume support. A full run is hours long on modest hardware and can be interrupted
    # by anything — an OOM kill, a reboot, Ctrl-C. Predictions are therefore appended as
    # each solution finishes, and a resumed run skips what is already on disk.
    predictions: list[SolutionPrediction] = []
    done_uids: set[str] = set()
    if cfg.run.resume and predictions_path.exists():
        with predictions_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                done_uids.add(row["uid"])
                predictions.append(
                    SolutionPrediction(
                        uid=row["uid"],
                        subset=row["subset"],
                        gold_label=row["gold_label"],
                        predicted_label=row["predicted_label"],
                        n_steps=row["n_steps"],
                        n_steps_scored=row["n_steps_scored"],
                        predicted_error_type=row["predicted_error_type"],
                        n_references_used=row["n_references_used"],
                    )
                )
        logger.info("Resuming: %d solution(s) already scored, skipping them", len(done_uids))

    pending = [s for s in solutions if s.uid not in done_uids]
    if not pending:
        logger.info("Nothing left to score — every solution is already in %s", predictions_path)

    mode = "a" if done_uids else "w"
    started = time.time()

    predictions_handle = predictions_path.open(mode, encoding="utf-8")
    traces_handle = traces_path.open(mode, encoding="utf-8") if cfg.run.save_traces else None

    try:
        for position, solution in enumerate(pending, start=1):
            prediction, traces = score_solution(solution, backend, retriever, cfg)
            predictions.append(prediction)

            # Flush every solution: an interrupted run must lose at most one.
            predictions_handle.write(json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n")
            predictions_handle.flush()

            if traces_handle is not None:
                for trace in traces:
                    traces_handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                traces_handle.flush()

            if position % cfg.run.log_every == 0 or position == len(pending):
                elapsed = time.time() - started
                accuracy = sum(p.correct for p in predictions) / len(predictions)
                remaining = (len(pending) - position) * elapsed / position
                logger.info(
                    "%d/%d  running acc %.3f  %.2fs/solution  eta %.1f min",
                    position,
                    len(pending),
                    accuracy,
                    elapsed / position,
                    remaining / 60,
                )
    finally:
        predictions_handle.close()
        if traces_handle is not None:
            traces_handle.close()

    by_subset: dict[str, tuple[list[int], list[int]]] = {}
    for subset, group in group_by_subset(solutions).items():
        uids = {s.uid for s in group}
        subset_predictions = [p for p in predictions if p.uid in uids and p.subset == subset]
        by_subset[subset] = (
            [p.gold_label for p in subset_predictions],
            [p.predicted_label for p in subset_predictions],
        )

    metrics = score_all(by_subset)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = {
        "run_name": cfg.run.name,
        "condition": "B (retrieval)" if cfg.retrieval.enabled else "A (baseline)",
        "environment": _environment(cfg),
        "n_solutions": len(predictions),
        "n_scored_this_session": len(pending),
        "n_resumed": len(done_uids),
        #: Wall time for THIS session only. On a resumed run it excludes earlier sessions,
        #: so do not use it for throughput figures unless n_resumed is 0.
        "wall_seconds": round(time.time() - started, 1),
        "metrics": metrics,
        "retrieval": retriever.stats.as_dict() if retriever else {"enabled": False},
        "dropped_references_for_length": getattr(backend, "dropped_references", 0),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if metrics["average_f1"] is None:
        logger.warning(
            "Average F1 is undefined: %s had no error cases or no clean cases. That is "
            "normal for a small --limit pilot and says nothing about model quality. "
            "Results are in %s",
            ", ".join(metrics["undefined_subsets"]) or "every subset",
            run_dir,
        )
    else:
        if metrics["undefined_subsets"]:
            logger.warning(
                "Excluded from the average (undefined F1, empty error or clean "
                "population): %s",
                ", ".join(metrics["undefined_subsets"]),
            )
        logger.info("Average F1 = %.4f  ->  %s", metrics["average_f1"], run_dir)
    return summary

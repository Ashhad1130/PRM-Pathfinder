"""Typed experiment configuration.

One YAML file fully determines a run. The resolved config is written into every run
directory as `config.resolved.yaml` so a result can always be traced back to the exact
settings that produced it — including the code version, see `eval.runner`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

import yaml

PROCESSBENCH_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")

# Ordered by out-of-distribution severity, per the project's expected-evidence table.
# Used by the analysis to test the "gains grow with difficulty" trend.
OOD_ORDER = {"gsm8k": 0, "math": 1, "olympiadbench": 2, "omnimath": 3}


@dataclass
class DataConfig:
    """Where the evaluation data and the retrieval pool come from."""

    processbench_id: str = "Qwen/ProcessBench"
    subsets: tuple[str, ...] = PROCESSBENCH_SUBSETS
    #: Cap examples per subset. None = full benchmark. Use a small number for pilots.
    limit_per_subset: int | None = None
    #: Local fixture directory; when set, ProcessBench is read from JSONL instead of the Hub.
    fixtures_dir: str | None = None

    pool_id: str = "declare-lab/PathFinder-600K"
    pool_split: str = "train"
    #: Cap pool size. The full 600K is unnecessary for a seminar-scale study; 50K is plenty.
    pool_max_items: int | None = 50_000
    pool_dir: str = "data/pool"


@dataclass
class RetrievalConfig:
    """RetrievalPRM's recipe: SBERT embedding -> PCA -> cosine similarity."""

    enabled: bool = True
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_batch_size: int = 128
    normalize: bool = True

    pca_components: int | None = 128
    pca_whiten: bool = False

    #: Stage 1 — question-level retrieval (RetrievalPRM's fix for dataset shift).
    top_k_questions: int = 8
    #: Stage 2 — step-level rerank within those questions (fix for reasoning-style shift).
    top_k_steps: int = 2
    #: Skip stage 2 and take steps straight from the top questions.
    step_level: bool = True

    #: Contamination guard: drop any pool question this similar to the eval question.
    #: ProcessBench and PathFinder-600K share upstream sources (MATH/GSM8K), so without
    #: this the "retrieved reference" can be the answer key. See docs/EXPERIMENTS.md.
    max_question_similarity: float = 0.95
    #: Also drop exact normalised string matches regardless of the threshold above.
    drop_exact_duplicates: bool = True

    index_dir: str = "artifacts/index"


@dataclass
class PRMConfig:
    """The frozen scoring contract. Changing anything here invalidates A-vs-B parity."""

    #: "mock" | "hf" | "hf4bit" | "int4"
    #:   hf     — full precision (needs ~15 GB VRAM)
    #:   hf4bit — NF4 via bitsandbytes
    #:   int4   — int4 weight-only via torchao; no bitsandbytes, works on Blackwell.
    #:            Fits a 7B in ~5.5 GB, which removes disk offload entirely.
    backend: str = "hf"

    #: int4 backend: quantisation group size. Smaller = more accurate, slightly larger.
    int4_group_size: int = 64
    #: int4 backend: torchao packing kernel. "tile_packed_to_4d" is the one that works on
    #: sm_120 (Blackwell); "plain" and "preshuffled" need the extra `mslk` package.
    int4_packing_format: str = "tile_packed_to_4d"
    model_name: str = "declare-lab/PathFinder-PRM-7B"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    #: "auto" picks flash_attention_2 when importable, else sdpa. Windows -> sdpa.
    attn_implementation: str = "auto"
    trust_remote_code: bool = True

    #: Per-device weight budget for accelerate, e.g. {0: "7GiB", "cpu": "10GiB"}.
    #: Needed whenever the model does not fit in VRAM. Keys are read from YAML as
    #: strings; integer-looking keys are converted back to device ordinals.
    max_memory: dict | None = None
    #: Directory for layers that fit in neither VRAM nor RAM. Disk offload works but is
    #: extremely slow — seconds per forward pass. Pilot runs only.
    offload_folder: str | None = None

    #: A step counts as erroneous when stage 1 emits <-> for math or consistency,
    #: or when the stage-2 correctness probability falls below this threshold.
    correctness_threshold: float = 0.5
    #: Stop scoring a solution at its first erroneous step (ProcessBench only needs the
    #: first error index). Set False to collect full per-step traces for error analysis.
    early_stop: bool = True

    max_input_tokens: int = 8192
    seed: int = 17


@dataclass
class PromptConfig:
    """How retrieved references are rendered into the user turn."""

    #: Keep verbatim from the model card, typo included — prompt fidelity matters.
    prompt_prefix: str = (
        "You are a Math Teacher. Given a question and a student's solution, evaluate the "
        "mathemetical correctness, logic consistency of the current step and whether it "
        "will lead to the correct final solution"
    )
    #: Show the reference step's gold Math/Consistency labels alongside it.
    include_reference_labels: bool = True
    #: Truncate each reference step to this many characters (keeps prompts bounded).
    max_reference_chars: int = 600


@dataclass
class RunConfig:
    name: str = "unnamed"
    out_dir: str = "runs"
    #: Write one JSONL record per scored solution, for later error analysis.
    save_traces: bool = True
    log_every: int = 25
    #: Skip solutions already present in the run's predictions.jsonl and append to it.
    #: A full run takes hours; without this, any interruption costs all of them.
    resume: bool = False


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    prm: PRMConfig = field(default_factory=PRMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)

    # ---- convenience -----------------------------------------------------------------

    @property
    def run_dir(self) -> Path:
        return Path(self.run.out_dir) / self.run.name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def _coerce(cls, payload: dict[str, Any]):
    """Build a dataclass from a dict, ignoring unknown keys loudly."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(payload) - set(known)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        # `subsets` is the only tuple-typed field; YAML gives us a list.
        kwargs[key] = tuple(value) if key == "subsets" and isinstance(value, list) else value
    return cls(**kwargs)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config, apply dotted-key overrides, and validate."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    if overrides:
        raw = copy.deepcopy(raw)
        for dotted, value in overrides.items():
            node = raw
            *parents, leaf = dotted.split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = value

    sections = {
        "run": RunConfig,
        "data": DataConfig,
        "retrieval": RetrievalConfig,
        "prm": PRMConfig,
        "prompt": PromptConfig,
    }
    unknown_sections = set(raw) - set(sections)
    if unknown_sections:
        raise ValueError(f"Unknown config section(s): {sorted(unknown_sections)}")

    cfg = Config(**{name: _coerce(cls, raw.get(name, {}) or {}) for name, cls in sections.items()})
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    bad = [s for s in cfg.data.subsets if s not in PROCESSBENCH_SUBSETS]
    if bad:
        raise ValueError(f"Unknown ProcessBench subset(s): {bad}. Valid: {PROCESSBENCH_SUBSETS}")

    if cfg.prm.backend not in {"mock", "hf", "hf4bit", "int4"}:
        raise ValueError(f"prm.backend must be mock|hf|hf4bit|int4, got {cfg.prm.backend!r}")

    if not 0.0 < cfg.prm.correctness_threshold < 1.0:
        raise ValueError("prm.correctness_threshold must lie strictly between 0 and 1")

    if cfg.retrieval.enabled:
        if cfg.retrieval.top_k_questions < 1:
            raise ValueError("retrieval.top_k_questions must be >= 1")
        if cfg.retrieval.top_k_steps < 1:
            raise ValueError("retrieval.top_k_steps must be >= 1")
        if not 0.0 < cfg.retrieval.max_question_similarity <= 1.0:
            raise ValueError("retrieval.max_question_similarity must lie in (0, 1]")

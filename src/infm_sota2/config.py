"""Einstellungen eines Laufs aus einer TOML-Datei in ``configs/``.

Jede Einstellung steht in der Datei, es gibt keine Defaults. Fehlende und
unbekannte Schlüssel, falsche Typen und unbekannte Werte brechen mit dem Namen des
Schlüssels ab. Für die Suche wählt ``with_grid`` die Kombination aus Lernrate und
Batchgröße zum Index des Job-Arrays.
"""

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, get_args

from infm_sota2.loop import TrainPrecision
from infm_sota2.model import GPT2_SMALL_CONFIG, MODEL_CLASSES


@dataclass(frozen=True)
class DataSettings:
    file: str
    start_line: str
    end_line: str
    book_title: str
    train_ratio: float
    val_ratio: float
    seq_len: int
    max_chars: dict[str, int] | None  # optionale Tabelle [data.max_chars]


@dataclass(frozen=True)
class TrainSettings:
    batch_size: int  # über alle GPUs zusammen
    learning_rate: float
    epochs: int
    weight_decay: float
    grad_clip: float
    eval_every: int
    precision: TrainPrecision
    compile_backend: str | None  # in TOML "none": nicht kompilieren
    fused_adamw: bool


@dataclass(frozen=True)
class GenerationSettings:
    prompt: str
    samples: int
    max_new_tokens: int
    temperature: float
    top_k: int


@dataclass(frozen=True)
class SearchSettings:
    learning_rates: tuple[float, ...]
    batch_sizes: tuple[int, ...]
    epochs: int


@dataclass(frozen=True)
class RunConfig:
    name: str
    group: str
    seed: int
    model_class: str
    evaluate_pretrained: bool
    save_checkpoint: bool
    report: bool
    data: DataSettings
    model: dict[str, int | float | bool]  # dieselben Schlüssel wie GPT2_SMALL_CONFIG
    train: TrainSettings
    generation: GenerationSettings
    search: SearchSettings | None


_TOP = {
    "name": str,
    "group": str,
    "seed": int,
    "model_class": str,
    "evaluate_pretrained": bool,
    "save_checkpoint": bool,
    "report": bool,
    "data": dict,
    "model": dict,
    "train": dict,
    "generation": dict,
    "search": dict,
}
_DATA = {
    "file": str,
    "start_line": str,
    "end_line": str,
    "book_title": str,
    "train_ratio": float,
    "val_ratio": float,
    "seq_len": int,
    "max_chars": dict,
}
_MAX_CHARS = {"train": int, "val": int, "test": int}
_MODEL = {key: type(value) for key, value in GPT2_SMALL_CONFIG.items()}
_TRAIN = {
    "batch_size": int,
    "learning_rate": float,
    "epochs": int,
    "weight_decay": float,
    "grad_clip": float,
    "eval_every": int,
    "precision": str,
    "compile_backend": str,
    "fused_adamw": bool,
}
_GENERATION = {
    "prompt": str,
    "samples": int,
    "max_new_tokens": int,
    "temperature": float,
    "top_k": int,
}
_SEARCH = {"learning_rates": list, "batch_sizes": list, "epochs": int}


def load_config(path: Path) -> RunConfig:
    with path.open("rb") as file:
        raw = tomllib.load(file)
    top = _table(raw, "", _TOP, optional={"search"})
    data = _table(top["data"], "data.", _DATA, optional={"max_chars"})
    max_chars = data.pop("max_chars", None)
    if max_chars is not None:
        max_chars = _table(max_chars, "data.max_chars.", _MAX_CHARS)
    model = _table(top["model"], "model.", _MODEL)
    train = _table(top["train"], "train.", _TRAIN)
    generation = _table(top["generation"], "generation.", _GENERATION)

    _one_of("model_class", top["model_class"], tuple(MODEL_CLASSES))
    _one_of("train.precision", train["precision"], get_args(TrainPrecision))
    if train["compile_backend"] == "none":
        train["compile_backend"] = None
    if data["seq_len"] > model["context_length"]:
        raise ValueError(
            f"data.seq_len {data['seq_len']} is longer than "
            f"model.context_length {model['context_length']}"
        )

    search = None
    if "search" in top:
        values = _table(top["search"], "search.", _SEARCH)
        search = SearchSettings(
            learning_rates=_numbers("search.learning_rates", values["learning_rates"]),
            batch_sizes=_integers("search.batch_sizes", values["batch_sizes"]),
            epochs=values["epochs"],
        )
    return RunConfig(
        name=top["name"],
        group=top["group"],
        seed=top["seed"],
        model_class=top["model_class"],
        evaluate_pretrained=top["evaluate_pretrained"],
        save_checkpoint=top["save_checkpoint"],
        report=top["report"],
        data=DataSettings(**data, max_chars=max_chars),
        model=model,
        train=TrainSettings(**train),
        generation=GenerationSettings(**generation),
        search=search,
    )


def with_grid(config: RunConfig, index: int) -> RunConfig:
    """Kombination Nr. ``index`` der Suche, zeilenweise über die Lernraten.

    Index 0 bis ``len(batch_sizes) - 1`` hat die erste Lernrate, dann die zweite.
    """
    if config.search is None:
        raise ValueError(f"{config.name}: a grid index needs a [search] table")
    learning_rates, batch_sizes = (
        config.search.learning_rates,
        config.search.batch_sizes,
    )
    size = len(learning_rates) * len(batch_sizes)
    if not 0 <= index < size:
        raise ValueError(f"grid index {index} is outside 0..{size - 1}")
    learning_rate = learning_rates[index // len(batch_sizes)]
    batch_size = batch_sizes[index % len(batch_sizes)]
    train = replace(
        config.train,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=config.search.epochs,
    )
    name = f"{config.name}-lr{learning_rate:g}-bs{batch_size}"
    return replace(config, name=name, train=train)


def _table(
    raw: dict[str, Any],
    prefix: str,
    spec: dict[str, type],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    """Prüft Schlüssel und Typen einer Tabelle, ``prefix`` steht vor jedem Namen."""
    missing = sorted(spec.keys() - raw.keys() - (optional or set()))
    if missing:
        names = ", ".join(prefix + key for key in missing)
        raise ValueError(f"missing setting(s): {names}")
    unknown = sorted(raw.keys() - spec.keys())
    if unknown:
        names = ", ".join(prefix + key for key in unknown)
        raise ValueError(f"unknown setting(s): {names}")
    return {key: _typed(prefix + key, raw[key], spec[key]) for key in raw}


def _typed(name: str, value: object, expected: type) -> Any:
    # bool ist in Python ein int, zählt hier aber weder als int noch als float
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name} must be a number, got {value!r}")
        return float(value)
    if expected is int and isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if not isinstance(value, expected):
        raise ValueError(f"{name} must be {expected.__name__}, got {value!r}")
    return value


def _numbers(name: str, values: list[object]) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    return tuple(_typed(f"{name}[{i}]", v, float) for i, v in enumerate(values))


def _integers(name: str, values: list[object]) -> tuple[int, ...]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    return tuple(_typed(f"{name}[{i}]", v, int) for i, v in enumerate(values))


def _one_of(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} is {value!r}, allowed: {', '.join(allowed)}")

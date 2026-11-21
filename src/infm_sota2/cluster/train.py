"""Smoke-Lauf und Läufe Nr. 2 bis 8: ein Trainingslauf nach einer TOML-Datei.

Liest die Konfiguration, baut Daten und Modell, misst auf Wunsch GPT-2 original,
trainiert mit ``Trainer.fit`` und schreibt eine JSON-Datei mit allen Einstellungen
und Ergebnissen nach ``results/week06/<benutzer>/<gruppe>/``.

Aufruf: ``python -m infm_sota2.cluster.train --config configs/smoke-cpu.toml``, auf
dem Cluster über ``slurm/train.sbatch``. Die Suche wählt mit ``--grid-index`` eine
Kombination aus Lernrate und Batchgröße.

Fassung für Schritt 2: Das Training kann nur, was der heutige ``Trainer`` kann,
also ein Prozess, ``fp32`` oder ``fp16-mixed``, ohne ``compile`` und ohne fused
AdamW. Alles andere bricht ab, bis der Umbau aus Schritt 3 fertig ist.
"""

import argparse
import getpass
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from infm_sota2.cluster.check import collect_info
from infm_sota2.cluster.config import (
    RunConfig,
    config_dict,
    load_config,
    run_label,
    with_grid,
)
from infm_sota2.cluster.results import git_commit, result_path, write_result
from infm_sota2.data import TextSplits
from infm_sota2.env import Environment
from infm_sota2.model import (
    GPT2_SMALL_CONFIG,
    GPT2_SMALL_WEIGHTS_SHA256,
    GPT2_SMALL_WEIGHTS_URL,
    GPTModel,
    GPTModelV2,
    build_model,
    download_weights,
    load_pretrained,
)
from infm_sota2.training import ModelReport, TrainConfig, Trainer

MODEL_CLASSES = {"GPTModel": GPTModel, "GPTModelV2": GPTModelV2}
PRETRAINED = "GPT-2 original"
PRETRAINED_FILE = "gpt2-small-124M.pth"
# Der Klon, aus dem dieser Code läuft (uv sync installiert das Paket editierbar).
# Sein Git-Commit kommt in jede Ergebnisdatei, egal aus welchem Ordner der Lauf kam.
REPO_DIR = Path(__file__).resolve().parents[3]


def run(
    config: RunConfig,
    env: Environment,
    user: str,
    job_id: str | None,
    repo_dir: Path,
) -> dict[str, object]:
    """Ein Lauf von den Daten bis zur JSON-Datei, gibt den Datensatz zurück.

    Bei ``FloatingPointError`` (Loss NaN) und ``torch.OutOfMemoryError`` landet
    der Lauf mit dem Fehler und ohne Loss in der Datei, danach geht der Fehler
    weiter. So bleibt ein gescheiterter Lauf in der Auswertung sichtbar.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_amp = _step2_use_amp(config, env, world_size)
    book = env.data_dir / config.data.file
    if not book.is_file():
        raise FileNotFoundError(
            f"{book} not found. `bash slurm/setup.sh` writes the mini book and lists "
            "what is missing; set SOTA2_DATA_DIR if the data lives in another folder."
        )
    data_fields = asdict(config.data)
    seq_len = data_fields.pop("seq_len")
    data = TextSplits.from_book(env.data_dir, seq_len=seq_len, **data_fields)

    label = run_label(config, world_size)
    context_length = config.model["context_length"]
    assert isinstance(context_length, int)  # load_config hat den Typ geprüft
    report = None
    if config.report or config.evaluate_pretrained:
        report = ModelReport(
            data,
            env.device,
            batch_size=config.train.batch_size,
            seed=config.seed,
            **asdict(config.generation),
        )
    record: dict[str, Any] = {
        "user": user,
        "label": label,
        "group": config.group,
        "config": config_dict(config),
        "environment": collect_info(env, check_internet=False)
        | {
            "job_id": job_id,
            "nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "world_size": world_size,
            "git_commit": git_commit(repo_dir),
        },
        "result": None,
        "test_loss": None,
        "perplexity": None,
        "texts": None,
        "copied": None,
        "pretrained": None,
        "peak_memory_gb": None,
        "fit_seconds": None,
        "error": None,
    }
    path = result_path(env.base_dir, user, config.group, label, job_id)
    try:
        if config.evaluate_pretrained:
            assert report is not None
            weights = download_weights(
                GPT2_SMALL_WEIGHTS_URL,
                GPT2_SMALL_WEIGHTS_SHA256,
                env.data_dir / PRETRAINED_FILE,
            )
            pretrained = load_pretrained(
                GPTModel, GPT2_SMALL_CONFIG, weights, env.device
            )
            pretrained_context = GPT2_SMALL_CONFIG["context_length"]  # 1024
            assert isinstance(pretrained_context, int)
            report.add(PRETRAINED, pretrained, pretrained_context)
            del pretrained
            record["pretrained"] = _report_entry(report, PRETRAINED)

        model = build_model(
            MODEL_CLASSES[config.model_class], config.model, config.seed, env.device
        )
        trainer = Trainer(
            env, wandb_config=config_dict(config) | {"user": user, "job_id": job_id}
        )
        train_config = TrainConfig(
            weight_decay=config.train.weight_decay,
            batch_size=config.train.batch_size,
            eval_every=config.train.eval_every,
            eval_batches=None,  # ganzer Validierungsteil wie in Woche 4
            grad_clip=config.train.grad_clip,
            use_amp=use_amp,
            seed=config.seed,
        )
        if env.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(env.device)
        start = time.perf_counter()
        result = trainer.fit(
            model,
            data,
            train_config,
            learning_rate=config.train.learning_rate,
            epochs=config.train.epochs,
            name=f"{label}-{user}",
            group=config.group,
            checkpoint_name=f"{label}.pth" if config.save_checkpoint else None,
        )
        record["fit_seconds"] = time.perf_counter() - start
        record["result"] = asdict(result)
        if env.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(env.device) / 1024**3
            record["peak_memory_gb"] = peak
        if config.report:
            assert report is not None
            report.add(label, model, context_length)
            record |= _report_entry(report, label)
    except (FloatingPointError, torch.OutOfMemoryError) as error:
        record |= {
            "error": f"{type(error).__name__}: {error}",
            "result": None,
            "test_loss": None,
            "perplexity": None,
            "texts": None,
            "copied": None,
            "fit_seconds": None,
        }
        write_result(path, record)
        print(f"Fehler, Ergebnis mit Fehler: {path}")
        raise
    write_result(path, record)
    print(f"Ergebnis: {path}")
    return record


def _step2_use_amp(config: RunConfig, env: Environment, world_size: int) -> bool:
    """Übersetzt ``precision`` für den heutigen Trainer (``use_amp``)."""
    later = []
    if config.train.precision == "bf16-mixed":
        later.append("precision bf16-mixed")
    if config.train.compile_backend is not None:
        later.append(f"compile_backend {config.train.compile_backend!r}")
    if config.train.fused_adamw:
        later.append("fused_adamw")
    if world_size > 1:
        later.append(f"WORLD_SIZE {world_size}")
    if later:
        raise ValueError(f"{', '.join(later)}: kommt mit Schritt 3 (Umbau nach Weg A)")
    if config.train.precision == "fp16-mixed" and env.device.type != "cuda":
        raise ValueError("fp16-mixed needs CUDA (GradScaler)")
    return config.train.precision == "fp16-mixed"


def _report_entry(report: ModelReport, name: str) -> dict[str, object]:
    """Test-Loss, Perplexity, Texte und längste kopierte Folgen eines Modells."""
    test_loss = report.test_loss(name)
    rows = [row for row in report.completions() if row["Modell"] == name]
    return {
        "test_loss": test_loss,
        "perplexity": math.exp(test_loss),
        "texts": [row["Text"] for row in rows],
        "copied": [row["Längste kopierte Folge"] for row in rows],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ein Trainingslauf nach einer TOML-Datei aus configs/"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--grid-index",
        type=int,
        help="Kombination der Suche, im Job-Array SLURM_ARRAY_TASK_ID",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.grid_index is not None:
        config = with_grid(config, args.grid_index)
    env = Environment.detect(smoke_test=False)
    run(config, env, getpass.getuser(), os.environ.get("SLURM_JOB_ID"), REPO_DIR)


if __name__ == "__main__":
    main()

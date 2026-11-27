import io
import json
import math
import os
import tempfile
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import TestCase, mock

import torch
from infm_sota2.config import RunConfig, TrainSettings, load_config, with_grid
from infm_sota2.train import fit, run

from infm_sota2.data import TextSplits
from infm_sota2.env import Environment
from infm_sota2.loop import TrainResult, evaluate, train
from infm_sota2.model import GPTModel
from tests.helpers import MINI_BOOK, TINY_CONFIG, local_env, tiny_model, write_mini_book

ROOT = Path(__file__).resolve().parent.parent
SMOKE_CPU = load_config(ROOT / "configs" / "smoke-cpu.toml")


def step2_config(config: RunConfig) -> RunConfig:
    """smoke-cpu.toml ohne die Optionen, die erst mit Schritt 3 kommen."""
    return replace(
        config, train=replace(config.train, precision="fp32", compile_backend=None)
    )


class TrainRunTest(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.env = local_env(self.base)
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("WORLD_SIZE", None)

    def tearDown(self):
        self.tmp.cleanup()

    def run_quietly(self, config: RunConfig) -> dict[str, Any]:
        with redirect_stdout(io.StringIO()):
            return run(config, self.env, "tester", None, ROOT)

    def written(self) -> list[dict[str, Any]]:
        folder = self.base / "results" / "week06" / "tester" / "week6-smoke"
        return [
            json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json")
        ]

    def test_smoke_run_without_wandb_writes_result(self):
        self.env.data_dir.mkdir()
        write_mini_book(self.env.data_dir)

        record = self.run_quietly(step2_config(SMOKE_CPU))

        files = self.written()
        self.assertEqual(len(files), 1)
        # JSON kennt keine Tupel, val_losses steht in der Datei als Liste
        self.assertEqual(files[0], json.loads(json.dumps(record)))
        self.assertIsNone(record["error"])
        # deutlich besser als Raten über 50.257 Tokens
        self.assertLess(record["result"]["best_val_loss"], math.log(50257) - 1)
        self.assertEqual(len(record["texts"]), 3)
        self.assertEqual(record["label"], "smoke-cpu-gpus1")
        self.assertRegex(record["environment"]["git_commit"], r"^[0-9a-f]{40}$")

    def test_record_keeps_its_keys(self):
        # Das Notebook von Woche 6 liest diese Schlüssel aus jeder Datei
        self.env.data_dir.mkdir()
        write_mini_book(self.env.data_dir)

        record = self.run_quietly(step2_config(SMOKE_CPU))

        keys = {
            "user",
            "label",
            "group",
            "config",
            "environment",
            "result",
            "mfu",
            "test_loss",
            "perplexity",
            "texts",
            "copied",
            "pretrained",
            "peak_memory_gb",
            "fit_seconds",
            "error",
        }
        self.assertEqual(set(record), keys)
        self.assertLessEqual(
            {"world_size", "gpu", "git_commit"}, set(record["environment"])
        )

    def test_missing_book_names_path_and_hint(self):
        with self.assertRaises(FileNotFoundError) as caught:
            self.run_quietly(step2_config(SMOKE_CPU))

        message = str(caught.exception)
        self.assertIn(str(self.env.data_dir / "mini-book.txt"), message)
        self.assertIn("SOTA2_DATA_DIR", message)
        self.assertIn("slurm/setup.sh", message)

    def test_fp16_on_cpu_is_an_error(self):
        config = replace(
            SMOKE_CPU,
            train=replace(
                SMOKE_CPU.train, precision="fp16-mixed", compile_backend=None
            ),
        )

        with self.assertRaisesRegex(ValueError, "CUDA"):
            self.run_quietly(config)

    def test_failed_training_writes_error_record(self):
        self.env.data_dir.mkdir()
        write_mini_book(self.env.data_dir)
        nan = FloatingPointError("loss is nan at step 3")

        with (
            mock.patch("infm_sota2.train.fit", side_effect=nan),
            self.assertRaises(FloatingPointError),
        ):
            self.run_quietly(step2_config(SMOKE_CPU))

        files = self.written()
        self.assertEqual(len(files), 1)
        self.assertIn("loss is nan", files[0]["error"])
        self.assertIsNone(files[0]["test_loss"])
        self.assertIsNone(files[0]["result"])

    def test_pretrained_model_is_measured_too(self):
        # Statt der OpenAI-Gewichte ein winziges Modell mit 1024 Tokens Kontext
        self.env.data_dir.mkdir()
        write_mini_book(self.env.data_dir)
        tiny = GPTModel({**TINY_CONFIG, "vocab_size": 50257, "context_length": 1024})
        config = replace(step2_config(SMOKE_CPU), evaluate_pretrained=True)

        with (
            mock.patch("infm_sota2.train.download_weights") as download,
            mock.patch("infm_sota2.train.load_pretrained", return_value=tiny),
        ):
            record = self.run_quietly(config)

        self.assertEqual(
            download.call_args.args[2], self.env.data_dir / "gpt2-small-124M.pth"
        )
        pretrained = record["pretrained"]
        self.assertEqual(len(pretrained["texts"]), 3)
        self.assertAlmostEqual(
            pretrained["perplexity"], math.exp(pretrained["test_loss"])
        )
        self.assertIsNone(record["error"])

    def test_smoke_run_with_bf16_and_compile(self):
        self.env.data_dir.mkdir()
        write_mini_book(self.env.data_dir)

        record = self.run_quietly(SMOKE_CPU)  # bf16-mixed und eager, unverändert

        self.assertIsNone(record["error"])
        val_losses = record["result"]["val_losses"]
        self.assertTrue(val_losses)
        self.assertLess(val_losses[-1][1], val_losses[0][1])
        self.assertEqual(record["environment"]["world_size"], 1)
        self.assertIsNone(record["mfu"])  # ohne bekannte GPU

    def test_out_of_memory_writes_error_record(self):
        self.env.data_dir.mkdir()
        write_mini_book(self.env.data_dir)
        oom = torch.OutOfMemoryError("CUDA out of memory (simuliert)")

        with (
            mock.patch("infm_sota2.train.fit", side_effect=oom),
            self.assertRaises(torch.OutOfMemoryError),
        ):
            self.run_quietly(SMOKE_CPU)

        files = self.written()
        self.assertEqual(len(files), 1)
        self.assertIn("out of memory", files[0]["error"])
        self.assertIsNone(files[0]["result"])

    def test_grid_run_uses_search_settings(self):
        self.env.data_dir.mkdir()
        write_mini_book(self.env.data_dir)
        search = (
            "\n[search]\nlearning_rates = [1e-3, 1e-2]\n"
            "batch_sizes = [2, 4]\nepochs = 1\n"
        )
        path = self.base / "search.toml"
        text = (ROOT / "configs" / "smoke-cpu.toml").read_text(encoding="utf-8")
        path.write_text(text + search, encoding="utf-8")

        record = self.run_quietly(with_grid(load_config(path), 1))

        names = [p.name for p in (self.base / "results").rglob("*.json")]
        self.assertEqual(names, ["smoke-cpu-lr0.001-bs4-gpus1-local.json"])
        self.assertEqual(record["config"]["train"]["epochs"], 1)
        self.assertEqual(record["config"]["train"]["batch_size"], 4)


CPU = torch.device("cpu")
SEED = 1
SETTINGS = TrainSettings(
    batch_size=4,
    learning_rate=1e-2,
    epochs=1,
    weight_decay=0.1,
    grad_clip=1.0,
    eval_every=5,
    precision="fp32",
    compile_backend=None,
    fused_adamw=False,
)


class FitCase(TestCase):
    """Mini-Buch und Daten, ohne eigene Tests."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        write_mini_book(base)
        self.env = local_env(base)
        # Gekürzt, damit die Tests schnell bleiben
        max_chars = {"train": 1500, "val": 400, "test": 400}
        self.data = TextSplits.from_book(
            base,
            **MINI_BOOK,
            train_ratio=0.8,
            val_ratio=0.1,
            max_chars=max_chars,
            seq_len=8,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def fit(
        self,
        model: torch.nn.Module,
        checkpoint_name: str | None,
        env: Environment | None = None,
        settings: TrainSettings = SETTINGS,
    ) -> TrainResult:
        with redirect_stdout(io.StringIO()):
            return fit(
                self.env if env is None else env,
                model,
                self.data,
                settings,
                SEED,
                name="test-run",
                group="tests",
                checkpoint_name=checkpoint_name,
                wandb_config={"test": True},
            )


class FitTest(FitCase):
    def test_fit_gives_same_result_as_train(self):
        model = tiny_model(seed=1, vocab_size=50257)
        direct = train(
            model,
            torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1),
            self.data.train_loader(4, seed=1, rank=0, world_size=1),
            self.data.loader("val", 4),
            CPU,
            epochs=1,
            eval_every=5,
            grad_clip=1.0,
            precision="fp32",
            checkpoint_path=None,
            checkpoint_metadata={},
            log=lambda metrics, step: None,
            eval_model=model,
        )

        result = self.fit(tiny_model(seed=1, vocab_size=50257), checkpoint_name=None)

        # Die gemessene Geschwindigkeit ist bei zwei Läufen nie gleich
        self.assertEqual(
            replace(result, tokens_per_second=None),
            replace(direct, tokens_per_second=None),
        )

    def test_fit_keeps_best_checkpoint(self):
        model = tiny_model(seed=1, vocab_size=50257)

        result = self.fit(model, checkpoint_name="best.pth")

        self.assertTrue((self.env.checkpoint_dir / "best.pth").exists())
        # Direkt an den bleibenden Ort, ohne Umweg über einen zweiten Ordner
        self.assertFalse((self.env.checkpoint_dir / "runs").exists())
        self.assertAlmostEqual(
            evaluate(model, self.data.loader("val", 4), CPU),
            result.best_val_loss,
            places=5,
        )

    def test_fit_without_checkpoint_writes_nothing(self):
        self.fit(tiny_model(seed=1, vocab_size=50257), checkpoint_name=None)

        written = list(self.env.checkpoint_dir.rglob("*"))
        self.assertEqual(written, [])


class FitOptionsTest(FitCase):
    """Loader, Batchgröße, compile und fused AdamW, auf der CPU."""

    def test_fit_uses_train_loader_of_its_rank(self):
        original = TextSplits.train_loader

        with mock.patch.object(
            TextSplits, "train_loader", autospec=True, side_effect=original
        ) as loader:
            self.fit(tiny_model(seed=1, vocab_size=50257), checkpoint_name=None)

        # Batch 4 pro Prozess, Seed 1, Rang 0 von 1
        loader.assert_called_once_with(self.data, 4, 1, 0, 1)

    def test_batch_size_must_divide_world_size(self):
        env = replace(self.env, world_size=3)

        with (
            mock.patch("infm_sota2.train.train") as train_spy,
            self.assertRaisesRegex(ValueError, "world_size"),
        ):
            self.fit(tiny_model(seed=1, vocab_size=50257), None, env=env)
        train_spy.assert_not_called()

    def test_fused_adamw_needs_cuda(self):
        settings = replace(SETTINGS, fused_adamw=True)

        with self.assertRaisesRegex(ValueError, "CUDA"):
            self.fit(tiny_model(seed=1, vocab_size=50257), None, settings=settings)

    def test_compile_and_bf16_on_cpu(self):
        settings = replace(SETTINGS, precision="bf16-mixed", compile_backend="eager")

        result = self.fit(tiny_model(seed=1, vocab_size=50257), None, settings=settings)

        self.assertTrue(math.isfinite(result.best_val_loss))

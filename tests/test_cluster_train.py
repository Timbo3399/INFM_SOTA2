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

from infm_sota2.cluster.config import RunConfig, load_config
from infm_sota2.cluster.train import run
from infm_sota2.model import GPTModel
from infm_sota2.training import Trainer
from tests.helpers import TINY_CONFIG, local_env, write_mini_book

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

    def test_missing_book_names_path_and_hint(self):
        with self.assertRaises(FileNotFoundError) as caught:
            self.run_quietly(step2_config(SMOKE_CPU))

        message = str(caught.exception)
        self.assertIn(str(self.env.data_dir / "mini-book.txt"), message)
        self.assertIn("SOTA2_DATA_DIR", message)
        self.assertIn("slurm/setup.sh", message)

    def test_step3_features_are_rejected_for_now(self):
        with self.assertRaisesRegex(ValueError, "Schritt 3"):
            self.run_quietly(SMOKE_CPU)  # bf16-mixed und eager

    def test_several_processes_are_rejected_for_now(self):
        os.environ["WORLD_SIZE"] = "2"

        with self.assertRaisesRegex(ValueError, "Schritt 3"):
            self.run_quietly(step2_config(SMOKE_CPU))

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
            mock.patch.object(Trainer, "fit", side_effect=nan),
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
            mock.patch("infm_sota2.cluster.train.download_weights") as download,
            mock.patch("infm_sota2.cluster.train.load_pretrained", return_value=tiny),
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

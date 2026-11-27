import io
import math
import tempfile
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import TestCase, mock

import torch

from infm_sota2.cluster.config import TrainSettings
from infm_sota2.cluster.train import fit
from infm_sota2.data import TextSplits
from infm_sota2.env import Environment
from infm_sota2.loop import TrainResult, evaluate, train
from tests.helpers import MINI_BOOK, local_env, tiny_model, write_mini_book

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
            mock.patch("infm_sota2.cluster.train.train") as train_spy,
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

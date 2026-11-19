import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase

import torch

from infm_sota2.data import TextSplits
from infm_sota2.env import Environment
from infm_sota2.training import TrainConfig, Trainer, evaluate, train
from tests.helpers import MINI_BOOK, tiny_model, write_mini_book

CPU = torch.device("cpu")
CONFIG = TrainConfig(
    weight_decay=0.1,
    batch_size=4,
    eval_every=5,
    eval_batches=None,
    grad_clip=1.0,
    use_amp=False,
    seed=1,
)


class TrainerTest(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        write_mini_book(base)
        self.env = Environment(False, base, base / "runs", CPU, "disabled")
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
        self.trainer = Trainer(self.env, wandb_config={"test": True})

    def tearDown(self):
        self.tmp.cleanup()

    def fit(self, model, checkpoint_name):
        with redirect_stdout(io.StringIO()):
            return self.trainer.fit(
                model,
                self.data,
                CONFIG,
                learning_rate=1e-2,
                epochs=1,
                name="test-run",
                group="tests",
                checkpoint_name=checkpoint_name,
            )

    def test_fit_gives_same_result_as_train(self):
        model = tiny_model(seed=1, vocab_size=50257)
        direct = train(
            model,
            torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1),
            self.data.train_loader(4, seed=1),
            self.data.loader("val", 4),
            CPU,
            epochs=1,
            eval_every=5,
            eval_batches=None,
            grad_clip=1.0,
            use_amp=False,
            checkpoint_path=None,
            checkpoint_metadata={},
            log=lambda metrics, step: None,
        )

        result = self.fit(tiny_model(seed=1, vocab_size=50257), checkpoint_name=None)

        self.assertEqual(result, direct)

    def test_fit_keeps_best_checkpoint(self):
        model = tiny_model(seed=1, vocab_size=50257)

        result = self.fit(model, checkpoint_name="best.pth")

        self.assertTrue((self.env.checkpoint_dir / "best.pth").exists())
        self.assertAlmostEqual(
            evaluate(model, self.data.loader("val", 4), CPU, None),
            result.best_val_loss,
            places=5,
        )

    def test_fit_without_checkpoint_writes_nothing(self):
        self.fit(tiny_model(seed=1, vocab_size=50257), checkpoint_name=None)

        written = list(self.env.checkpoint_dir.rglob("*"))
        self.assertEqual(written, [])

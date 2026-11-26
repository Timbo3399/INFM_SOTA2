import math
import tempfile
from pathlib import Path
from unittest import TestCase

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from infm_sota2.training import (
    batch_loss,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train,
)
from tests.helpers import TINY_CONFIG, pattern_loader, tiny_model

CPU = torch.device("cpu")


class NanModel(nn.Module):
    """Liefert NaN-Logits, um den Abbruch bei kaputtem Loss zu prüfen."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.full((*x.shape, 20), float("nan")) * self.weight


class EvaluateTest(TestCase):
    def test_untrained_loss_is_close_to_log_vocab(self):
        # Ein untrainiertes Modell rät: Loss ≈ ln(vocab_size)
        loss = evaluate(tiny_model(seed=123), pattern_loader(16, 4), CPU, None)

        self.assertAlmostEqual(loss, math.log(TINY_CONFIG["vocab_size"]), delta=0.5)

    def test_evaluate_respects_max_batches(self):
        model = tiny_model(seed=123)
        loader = pattern_loader(16, 4)
        inputs, targets = next(iter(loader))

        loss = evaluate(model, loader, CPU, max_batches=1)

        with torch.no_grad():
            expected = batch_loss(model.eval(), inputs, targets).item()
        self.assertAlmostEqual(loss, expected, places=6)

    def test_evaluate_weights_batches_by_token_count(self):
        model = tiny_model(seed=123)
        loader = pattern_loader(5, 4)  # Batches mit 4 und 1 Fenster
        all_inputs = torch.cat([x for x, _ in loader])
        all_targets = torch.cat([y for _, y in loader])

        loss = evaluate(model, loader, CPU, None)

        with torch.no_grad():
            expected = batch_loss(model.eval(), all_inputs, all_targets).item()
        self.assertAlmostEqual(loss, expected, places=5)

    def test_evaluate_restores_training_mode(self):
        model = tiny_model(seed=123).train()

        evaluate(model, pattern_loader(4, 4), CPU, None)

        self.assertTrue(model.training)


class TrainTest(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.checkpoint = Path(self.tmp.name) / "best.pth"
        self.logs: list[tuple[dict[str, float], int]] = []

    def tearDown(self):
        self.tmp.cleanup()

    def run_train(self, model, **kwargs):
        args = {
            "optimizer": torch.optim.AdamW(model.parameters(), lr=1e-2),
            "train_loader": pattern_loader(16, 4),
            "val_loader": pattern_loader(4, 4),
            "device": CPU,
            "epochs": 1,
            "eval_every": 2,
            "eval_batches": None,
            "grad_clip": 1.0,
            "precision": "fp32",
            "checkpoint_path": None,
            "checkpoint_metadata": {},
            "log": lambda metrics, step: self.logs.append((metrics, step)),
            "eval_model": model,
        }
        return train(model, **{**args, **kwargs})

    def test_training_lowers_validation_loss(self):
        model = tiny_model(seed=123)
        before = evaluate(model, pattern_loader(4, 4), CPU, None)

        result = self.run_train(model, epochs=3)

        self.assertLess(result.best_val_loss, before - 1.0)

    def test_logs_train_and_val_loss(self):
        result = self.run_train(tiny_model(seed=123))

        # 16 Fenster / Batch 4 = 4 Schritte, Prüfung alle 2 Schritte
        self.assertEqual([step for _, step in self.logs], [2, 4])
        for metrics, _ in self.logs:
            self.assertEqual(
                set(metrics),
                {
                    "train/loss",
                    "val/loss",
                    "epoch",
                    "tokens_seen",
                    "throughput/tokens_per_second",
                },
            )
        self.assertEqual(self.logs[-1][0]["tokens_seen"], 16 * 8)
        self.assertEqual(self.logs[-1][0]["epoch"], 1.0)
        self.assertEqual(result.steps, 4)
        self.assertEqual(result.tokens_seen, 16 * 8)

    def test_evaluates_at_end_of_epoch(self):
        self.run_train(tiny_model(seed=123), epochs=2, eval_every=3)

        self.assertEqual([step for _, step in self.logs], [3, 4, 6, 8])

    def test_saves_best_checkpoint(self):
        result = self.run_train(
            tiny_model(seed=123),
            epochs=3,
            checkpoint_path=self.checkpoint,
            checkpoint_metadata={"run": "test"},
        )

        restored = tiny_model(seed=999)
        metadata = load_checkpoint(self.checkpoint, restored)
        self.assertEqual(metadata["run"], "test")
        self.assertEqual(metadata["step"], result.best_step)
        self.assertEqual(metadata["val_loss"], result.best_val_loss)
        self.assertAlmostEqual(
            evaluate(restored, pattern_loader(4, 4), CPU, None),
            result.best_val_loss,
            places=5,
        )

    def test_best_val_loss_is_minimum_of_logged_val_losses(self):
        result = self.run_train(tiny_model(seed=123), epochs=3)

        logged = [metrics["val/loss"] for metrics, _ in self.logs]
        self.assertEqual(result.best_val_loss, min(logged))
        self.assertEqual(result.best_step, self.logs[logged.index(min(logged))][1])

    def test_non_finite_loss_raises(self):
        with self.assertRaisesRegex(FloatingPointError, "step 0"):
            self.run_train(NanModel())

    def test_fp16_mixed_requires_cuda(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            self.run_train(tiny_model(seed=123), precision="fp16-mixed")

    def test_unknown_precision_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "bf16"):
            self.run_train(tiny_model(seed=123), precision="bf16")

    def test_bf16_mixed_trains_on_cpu(self):
        model = tiny_model(seed=123)
        before = evaluate(model, pattern_loader(4, 4), CPU, None)

        result = self.run_train(model, epochs=3, precision="bf16-mixed")

        self.assertLess(result.best_val_loss, before - 1.0)

    def test_checkpoint_from_compiled_model_loads_into_plain_model(self):
        model = tiny_model(seed=123)
        compiled = torch.compile(model, backend="eager")

        self.run_train(
            compiled,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-2),
            epochs=2,
            checkpoint_path=self.checkpoint,
            eval_model=model,
        )

        state = torch.load(self.checkpoint, weights_only=True)["model_state_dict"]
        self.assertFalse([key for key in state if key.startswith("_orig_mod.")])
        tiny_model(seed=999).load_state_dict(state)  # strikt

    def test_distributed_sampler_gets_each_epoch(self):
        epochs_seen = []

        class RecordingSampler(DistributedSampler):
            def set_epoch(self, epoch):
                epochs_seen.append(epoch)
                super().set_epoch(epoch)

        dataset = pattern_loader(16, 4).dataset
        sampler = RecordingSampler(dataset, num_replicas=1, rank=0, seed=1)
        loader = DataLoader(dataset, batch_size=4, sampler=sampler, drop_last=True)

        self.run_train(tiny_model(seed=123), train_loader=loader, epochs=3)

        self.assertEqual(epochs_seen, [0, 1, 2])

    def test_tokens_per_second_needs_two_intervals(self):
        # Das erste Intervall zählt nicht, dort kompiliert torch.compile
        only_end = self.run_train(tiny_model(seed=123), eval_every=100)
        every_step = self.run_train(tiny_model(seed=123), eval_every=1)

        self.assertIsNone(only_end.tokens_per_second)
        assert every_step.tokens_per_second is not None
        self.assertGreater(every_step.tokens_per_second, 0)

    def test_val_losses_match_logged_losses(self):
        result = self.run_train(tiny_model(seed=123), epochs=2)

        logged = tuple((step, metrics["val/loss"]) for metrics, step in self.logs)
        self.assertEqual(result.val_losses, logged)


class CheckpointTest(TestCase):
    def test_checkpoint_round_trip_restores_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "model.pth"
            original = tiny_model(seed=1)
            restored = tiny_model(seed=2)

            save_checkpoint(path, original, {"lr": 4e-4})
            metadata = load_checkpoint(path, restored)

        self.assertEqual(metadata, {"lr": 4e-4})
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)

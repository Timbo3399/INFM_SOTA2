import tempfile
from pathlib import Path
from unittest import TestCase

import torch

from infm_sota2 import GPTModel
from infm_sota2.model import build_model, load_pretrained
from tests.helpers import TINY_CONFIG, tiny_model

CPU = torch.device("cpu")


class BuildModelTest(TestCase):
    def test_build_model_is_seeded(self):
        first = build_model(GPTModel, TINY_CONFIG, seed=1, device=CPU)
        second = build_model(GPTModel, TINY_CONFIG, seed=1, device=CPU)

        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)


class LoadPretrainedTest(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "weights.pth"
        self.original = tiny_model(seed=1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_pretrained_restores_weights(self):
        torch.save(self.original.state_dict(), self.path)

        loaded = load_pretrained(GPTModel, TINY_CONFIG, self.path, CPU)

        for key, value in self.original.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)

    def test_load_pretrained_is_strict(self):
        state = self.original.state_dict()
        del state["out_head.weight"]
        torch.save(state, self.path)

        with self.assertRaisesRegex(RuntimeError, "out_head.weight"):
            load_pretrained(GPTModel, TINY_CONFIG, self.path, CPU)

    def test_load_pretrained_uses_rng_like_plain_construction(self):
        # Wie der alte Notebook-Code: kein eigener Seed, damit das Nachtrainieren
        # danach dieselben Zufallszahlen bekommt
        torch.save(self.original.state_dict(), self.path)
        torch.manual_seed(5)
        GPTModel(TINY_CONFIG)
        expected = torch.rand(1)

        torch.manual_seed(5)
        load_pretrained(GPTModel, TINY_CONFIG, self.path, CPU)

        self.assertTrue(torch.equal(torch.rand(1), expected))

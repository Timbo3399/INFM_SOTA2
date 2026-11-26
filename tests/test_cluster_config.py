import json
import tempfile
from pathlib import Path
from unittest import TestCase

from infm_sota2.cluster.config import (
    config_dict,
    load_config,
    run_label,
    with_grid,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
SMOKE_CPU = (CONFIGS / "smoke-cpu.toml").read_text(encoding="utf-8")
SEARCH = """
[search]
learning_rates = [1e-4, 3e-4, 1e-3, 3e-3]
batch_sizes = [8, 32, 128]
epochs = 1
"""


class ConfigTestCase(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, text: str):
        path = Path(self.tmp.name) / "config.toml"
        path.write_text(text, encoding="utf-8")
        return load_config(path)

    def changed(self, old: str, new: str) -> str:
        """smoke-cpu.toml mit genau einer Änderung."""
        self.assertEqual(SMOKE_CPU.count(old), 1, old)
        return SMOKE_CPU.replace(old, new)


class LoadConfigTest(ConfigTestCase):
    def test_loads_smoke_cpu_config(self):
        config = load_config(CONFIGS / "smoke-cpu.toml")

        self.assertEqual(config.name, "smoke-cpu")
        self.assertEqual(config.train.compile_backend, "eager")
        self.assertIs(config.train.fused_adamw, False)
        self.assertEqual(config.model["n_layers"], 2)
        self.assertIsNone(config.search)
        self.assertIsNone(config.data.max_chars)

    def test_every_config_file_loads(self):
        paths = sorted(CONFIGS.glob("*.toml"))
        self.assertGreaterEqual(len(paths), 3)
        for path in paths:
            with self.subTest(config=path.name):
                load_config(path)

    def test_missing_key_is_named(self):
        with self.assertRaisesRegex(ValueError, "epochs"):
            self.load(self.changed("epochs = 2\n", ""))

    def test_unknown_key_is_named(self):
        text = self.changed("[train]\n", "[train]\nepoch = 2\n")

        with self.assertRaisesRegex(ValueError, r"unknown.*\bepoch\b"):
            self.load(text)

    def test_wrong_type_is_an_error(self):
        cases = {
            "batch_size": ("batch_size = 4", "batch_size = true"),
            "learning_rate": ("learning_rate = 1e-2\n", 'learning_rate = "4e-4"\n'),
        }
        for key, (old, new) in cases.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.load(self.changed(old, new))

    def test_integer_is_accepted_as_float(self):
        config = self.load(self.changed("grad_clip = 1.0\n", "grad_clip = 1\n"))

        self.assertEqual(config.train.grad_clip, 1.0)
        self.assertIsInstance(config.train.grad_clip, float)

    def test_unknown_values_are_named(self):
        cases = {
            "precision": ('precision = "bf16-mixed"', 'precision = "bf16"'),
            "model_class": ('model_class = "GPTModelV2"', 'model_class = "GPT"'),
        }
        for key, (old, new) in cases.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.load(self.changed(old, new))

    def test_model_needs_every_key_of_gpt2_config(self):
        with self.assertRaisesRegex(ValueError, "qkv_bias"):
            self.load(self.changed("qkv_bias = true\n", ""))

    def test_seq_len_longer_than_context_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "seq_len"):
            self.load(self.changed("seq_len = 32\n", "seq_len = 64\n"))

    def test_compile_backend_none_means_no_compile(self):
        text = self.changed('compile_backend = "eager"', 'compile_backend = "none"')

        self.assertIsNone(self.load(text).train.compile_backend)

    def test_max_chars_table_is_optional(self):
        table = "\n[data.max_chars]\ntrain = 300\nval = 60\ntest = 60\n"
        text = self.changed("seq_len = 32\n", "seq_len = 32\n" + table)

        config = self.load(text)

        self.assertEqual(config.data.max_chars, {"train": 300, "val": 60, "test": 60})


class GridTest(ConfigTestCase):
    def test_grid_is_row_major(self):
        config = self.load(SMOKE_CPU + SEARCH)
        expected = {0: (1e-4, 8), 1: (1e-4, 32), 3: (3e-4, 8), 11: (3e-3, 128)}

        for index, (learning_rate, batch_size) in expected.items():
            with self.subTest(index=index):
                combination = with_grid(config, index)
                self.assertEqual(combination.train.learning_rate, learning_rate)
                self.assertEqual(combination.train.batch_size, batch_size)
                self.assertEqual(combination.train.epochs, 1)
        self.assertTrue(with_grid(config, 3).name.endswith("-lr0.0003-bs8"))

    def test_grid_index_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "12"):
            with_grid(self.load(SMOKE_CPU + SEARCH), 12)
        with self.assertRaisesRegex(ValueError, "search"):
            with_grid(self.load(SMOKE_CPU), 0)


class LabelTest(ConfigTestCase):
    def test_run_label_names_gpus(self):
        self.assertEqual(run_label(self.load(SMOKE_CPU), 2), "smoke-cpu-gpus2")

    def test_config_dict_is_ready_for_json(self):
        record = json.loads(json.dumps(config_dict(self.load(SMOKE_CPU + SEARCH))))

        self.assertEqual(record["train"]["learning_rate"], 1e-2)
        self.assertEqual(record["search"]["batch_sizes"], [8, 32, 128])

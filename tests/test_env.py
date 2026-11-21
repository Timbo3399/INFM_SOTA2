import os
from pathlib import Path
from unittest import TestCase, mock

import torch

from infm_sota2.env import Environment, version_mismatches


def clean_environment(test: TestCase) -> None:
    """Entfernt W&B-Modus und Datenordner der Person, die die Tests ausführt."""
    patcher = mock.patch.dict(os.environ)
    patcher.start()
    test.addCleanup(patcher.stop)
    for key in ("WANDB_MODE", "SOTA2_DATA_DIR"):
        os.environ.pop(key, None)


class EnvironmentTest(TestCase):
    def setUp(self):
        clean_environment(self)

    def test_detect_local_uses_working_directory(self):
        env = Environment.detect(smoke_test=True)

        self.assertFalse(env.in_colab)
        self.assertEqual(env.base_dir, Path.cwd())
        self.assertEqual(env.run_dir, Path.cwd() / "checkpoints" / "runs")
        self.assertEqual(env.data_dir, Path.cwd() / "data")
        self.assertEqual(env.checkpoint_dir, Path.cwd() / "checkpoints")

    def test_smoke_test_disables_wandb(self):
        self.assertEqual(Environment.detect(smoke_test=True).wandb_mode, "disabled")
        self.assertEqual(Environment.detect(smoke_test=False).wandb_mode, "online")

    def test_describe_names_device_and_torch(self):
        env = Environment.detect(smoke_test=True)

        description = env.describe()

        self.assertEqual(set(description), {"device", "gpu", "torch", "colab"})
        self.assertEqual(description["device"], env.device.type)
        self.assertEqual(description["torch"], torch.__version__)


class EnvironmentVariablesTest(TestCase):
    def setUp(self):
        clean_environment(self)

    def test_wandb_mode_from_environment(self):
        os.environ["WANDB_MODE"] = "offline"

        self.assertEqual(Environment.detect(smoke_test=False).wandb_mode, "offline")
        self.assertEqual(Environment.detect(smoke_test=True).wandb_mode, "disabled")

    def test_invalid_wandb_mode_is_an_error(self):
        os.environ["WANDB_MODE"] = "on"

        with self.assertRaisesRegex(ValueError, "WANDB_MODE"):
            Environment.detect(smoke_test=False)

    def test_data_dir_from_environment(self):
        os.environ["SOTA2_DATA_DIR"] = "/shared/sota2-data"

        env = Environment.detect(smoke_test=True)

        self.assertEqual(env.data_dir, Path("/shared/sota2-data"))

    def test_data_dir_defaults_to_working_directory(self):
        self.assertEqual(
            Environment.detect(smoke_test=True).data_dir, Path.cwd() / "data"
        )

    def test_empty_variables_count_as_unset(self):
        # wie ${VAR:-...} in slurm/setup.sh
        os.environ["WANDB_MODE"] = ""
        os.environ["SOTA2_DATA_DIR"] = ""

        env = Environment.detect(smoke_test=False)

        self.assertEqual(env.wandb_mode, "online")
        self.assertEqual(env.data_dir, Path.cwd() / "data")


class VersionMismatchTest(TestCase):
    def test_version_mismatch_is_reported(self):
        installed = {"torch": "2.14.1+cu130", "wandb": "0.27.0"}

        messages = version_mismatches(
            ["torch==2.14.1", "wandb==0.28.1"], installed.__getitem__
        )

        self.assertEqual(
            messages,
            ["Achtung: wandb 0.27.0 installiert, pyproject.toml erwartet 0.28.1"],
        )

    def test_matching_versions_give_no_message(self):
        installed = {"torch": "2.14.1+cpu"}

        messages = version_mismatches(["torch==2.14.1"], installed.__getitem__)

        self.assertEqual(messages, [])

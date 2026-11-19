from pathlib import Path
from unittest import TestCase

import torch

from infm_sota2.env import Environment, version_mismatches


class EnvironmentTest(TestCase):
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

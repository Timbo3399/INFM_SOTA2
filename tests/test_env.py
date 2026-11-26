import os
from pathlib import Path
from unittest import TestCase, mock

from infm_sota2.env import Environment


def clean_environment(test: TestCase) -> None:
    """Entfernt W&B-Modus, Datenordner und torchrun-Variablen der Umgebung, in
    der die Tests laufen."""
    patcher = mock.patch.dict(os.environ)
    patcher.start()
    test.addCleanup(patcher.stop)
    for key in ("WANDB_MODE", "SOTA2_DATA_DIR", "RANK", "LOCAL_RANK", "WORLD_SIZE"):
        os.environ.pop(key, None)


class EnvironmentTest(TestCase):
    def setUp(self):
        clean_environment(self)

    def test_detect_local_uses_working_directory(self):
        env = Environment.detect(smoke_test=True)

        self.assertEqual(env.base_dir, Path.cwd())
        self.assertEqual(env.data_dir, Path.cwd() / "data")
        self.assertEqual(env.checkpoint_dir, Path.cwd() / "checkpoints")

    def test_smoke_test_disables_wandb(self):
        self.assertEqual(Environment.detect(smoke_test=True).wandb_mode, "disabled")
        self.assertEqual(Environment.detect(smoke_test=False).wandb_mode, "online")


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

    def test_detect_reads_torchrun_variables(self):
        os.environ.update(RANK="1", LOCAL_RANK="0", WORLD_SIZE="2")

        env = Environment.detect(smoke_test=True)

        self.assertEqual((env.rank, env.world_size), (1, 2))

    def test_detect_without_torchrun_is_single_process(self):
        env = Environment.detect(smoke_test=True)

        self.assertEqual((env.rank, env.world_size), (0, 1))

    def test_partial_torchrun_variables_are_an_error(self):
        os.environ["RANK"] = "0"

        with self.assertRaisesRegex(ValueError, "WORLD_SIZE"):
            Environment.detect(smoke_test=True)

    def test_empty_variables_count_as_unset(self):
        # wie ${VAR:-...} in slurm/setup.sh
        os.environ["WANDB_MODE"] = ""
        os.environ["SOTA2_DATA_DIR"] = ""

        env = Environment.detect(smoke_test=False)

        self.assertEqual(env.wandb_mode, "online")
        self.assertEqual(env.data_dir, Path.cwd() / "data")

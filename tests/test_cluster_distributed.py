import tempfile
from pathlib import Path
from unittest import TestCase

import torch.distributed as dist

from infm_sota2.cluster.distributed import destroy_distributed, init_distributed
from tests.helpers import local_env


class DistributedTest(TestCase):
    def test_single_process_starts_no_process_group(self):
        # Ein Prozess ist die Basis für den Speedup und läuft ohne DDP
        with tempfile.TemporaryDirectory() as tmp:
            env = local_env(Path(tmp))

            init_distributed(env)
            started = dist.is_initialized()
            destroy_distributed(env)

        self.assertFalse(started)

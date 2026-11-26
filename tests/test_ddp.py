"""DDP rechnet richtig: zwei CPU-Prozesse mit Gloo stehen für zwei Knoten.

Zwei Prozesse mit je Batch 4 müssen nach einer Epoche dieselben Gewichte haben
wie ein Prozess mit Batch 8. Ohne Dropout sind die Updates bis auf die Rundung
gleich, weil DDP die Gradienten mittelt und der DistributedSampler bei gleicher
Gesamt-Batchgröße in jedem Schritt dieselben Fenster verteilt.
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

from infm_sota2.cluster.config import TrainSettings, load_config
from infm_sota2.cluster.train import fit, run
from infm_sota2.data import TextSplits
from tests.helpers import MINI_BOOK, local_env, tiny_model, write_mini_book

ROOT = Path(__file__).resolve().parent.parent
WORLD_SIZE = 2
SETTINGS = TrainSettings(
    batch_size=8,  # über beide Prozesse zusammen
    learning_rate=1e-2,
    epochs=1,
    weight_decay=0.1,
    grad_clip=1.0,
    eval_every=5,
    precision="fp32",
    compile_backend=None,
    fused_adamw=False,
)


def train_once(base: Path, data_dir: Path, rank: int, world_size: int) -> None:
    """Trainiert wie ein Prozess unter torchrun und speichert die Gewichte danach."""
    base.mkdir(parents=True, exist_ok=True)
    env = replace(local_env(base), rank=rank, world_size=world_size)
    data = TextSplits.from_book(
        data_dir,
        **MINI_BOOK,
        train_ratio=0.8,
        val_ratio=0.1,
        max_chars={"train": 1500, "val": 400, "test": 400},  # fünf Schritte
        seq_len=8,
    )
    # Ohne Dropout (TINY_CONFIG) und ohne Bias in Query, Key und Value: Der Gradient
    # des Key-Bias ist exakt null, weil Softmax eine Verschiebung aller Scores um
    # q·b nicht sieht. AdamW macht aus dem Rundungsrauschen darin volle Updates,
    # und das Rauschen hängt bei DDP von der Reihenfolge der Summen ab.
    model = tiny_model(seed=1, vocab_size=50257, qkv_bias=False)
    with redirect_stdout(io.StringIO()):
        fit(env, model, data, SETTINGS, 1, "ddp", "tests", "best.pth", {"test": True})
    torch.save(model.state_dict(), base / "final.pt")


def worker(rank: int, root: str) -> None:
    """Ein Prozess von zwei, mit eigenem Ordner unter ``root``."""
    folder = Path(root)
    rendezvous = (folder / "rendezvous").as_uri()
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=WORLD_SIZE
    )
    try:
        train_once(folder / f"rank{rank}", folder / "data", rank, WORLD_SIZE)
        smoke_run(folder / f"smoke{rank}", folder / "data", rank, WORLD_SIZE)
    finally:
        dist.destroy_process_group()


def smoke_run(base: Path, data_dir: Path, rank: int, world_size: int) -> None:
    """Der Smoke-Lauf aus configs/smoke-cpu.toml, wie unter torchrun."""
    base.mkdir(parents=True, exist_ok=True)
    env = replace(
        local_env(base),
        data_dir=data_dir,
        rank=rank,
        world_size=world_size,
    )
    # smoke-cpu.toml mit bf16, compile und Report, gekürzt auf eine kurze Epoche
    config = load_config(ROOT / "configs" / "smoke-cpu.toml")
    config = replace(
        config,
        data=replace(config.data, max_chars={"train": 3000, "val": 600, "test": 600}),
        train=replace(config.train, epochs=1),
        generation=replace(config.generation, samples=1, max_new_tokens=5),
    )
    with redirect_stdout(io.StringIO()):
        record = run(config, env, "tester", None, ROOT)
    (base / "returned.json").write_text(json.dumps(record), encoding="utf-8")


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "braucht Gloo")
class DistributedDataParallelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Einmal für beide Tests: zwei Prozesse, danach ein einzelner zum Vergleich
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        (cls.root / "data").mkdir()
        write_mini_book(cls.root / "data")
        spawn(worker, args=(cls.tmp.name,), nprocs=WORLD_SIZE, join=True)
        train_once(cls.root / "single", cls.root / "data", rank=0, world_size=1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_two_processes_match_one_process(self):
        # Rang 0 und der einzelne Prozess laden danach beide ihren besten Stand
        ddp = torch.load(self.root / "rank0" / "final.pt", weights_only=True)
        single = torch.load(self.root / "single" / "final.pt", weights_only=True)

        self.assertEqual(ddp.keys(), single.keys())
        for key, value in single.items():
            with self.subTest(key=key):
                self.assertTrue(torch.allclose(ddp[key], value, atol=1e-5))

    def test_only_rank_zero_writes_checkpoint(self):
        rank0 = self.root / "rank0" / "checkpoints"

        self.assertEqual([path.name for path in rank0.glob("*.pth")], ["best.pth"])
        self.assertFalse((self.root / "rank1" / "checkpoints").exists())

    def test_two_process_smoke_run_writes_one_file(self):
        files = sorted(self.root.glob("smoke*/results/**/*.json"))
        returned = [
            json.loads((self.root / f"smoke{rank}" / "returned.json").read_text())
            for rank in range(WORLD_SIZE)
        ]

        # Ordner des Rangs, der die Datei geschrieben hat
        writers = [path.relative_to(self.root).parts[0] for path in files]
        self.assertEqual(writers, ["smoke0"])
        self.assertEqual(returned[1], {})
        self.assertEqual(returned[0]["environment"]["world_size"], 2)
        self.assertIsNone(returned[0]["error"])

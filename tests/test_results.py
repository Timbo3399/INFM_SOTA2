import json
import re
import tempfile
from pathlib import Path
from unittest import TestCase

from infm_sota2.results import (
    git_commit,
    load_results,
    result_path,
    scaling_table,
    search_table,
    write_result,
)

ROOT = Path(__file__).resolve().parent.parent


def scaling_record(world_size: int, tokens_per_second: float | None, error=None):
    return {
        "environment": {"world_size": world_size},
        "result": None if error else {"tokens_per_second": tokens_per_second},
        "error": error,
        "file": f"scaling-gpus{world_size}.json",
    }


def search_record(learning_rate: float, batch_size: int, loss=None, error=None):
    return {
        "config": {"train": {"learning_rate": learning_rate, "batch_size": batch_size}},
        "result": None if error else {"best_val_loss": loss},
        "error": error,
        "file": f"search-lr{learning_rate:g}-bs{batch_size}.json",
    }


class ResultFileTest(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_result_path_layout(self):
        week = self.base / "results" / "week06" / "tester" / "week6-smoke"

        self.assertEqual(
            result_path(self.base, "tester", "week6-smoke", "smoke-gpus1", "42"),
            week / "smoke-gpus1-job42.json",
        )
        self.assertEqual(
            result_path(self.base, "tester", "week6-smoke", "smoke-gpus1", None),
            week / "smoke-gpus1-local.json",
        )

    def test_write_refuses_to_overwrite(self):
        path = self.base / "a" / "b.json"
        write_result(path, {"x": 1})

        with self.assertRaises(FileExistsError):
            write_result(path, {"x": 2})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"x": 1})

    def test_unserializable_record_leaves_no_file(self):
        path = self.base / "kaputt.json"

        with self.assertRaises(TypeError):
            write_result(path, {"x": object()})
        self.assertFalse(path.exists())

    def test_roundtrip_keeps_record(self):
        record = {"user": "tester", "text": "Elben über Mittelerde", "loss": 4.25}
        write_result(self.base / "g" / "lauf-job1.json", record)

        loaded = load_results(self.base)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(
            loaded[0], record | {"file": str(self.base / "g" / "lauf-job1.json")}
        )

    def test_git_commit_is_hex(self):
        self.assertRegex(git_commit(ROOT), r"^[0-9a-f]{40}$")


class ScalingTableTest(TestCase):
    def test_scaling_table_by_hand(self):
        records = [
            scaling_record(4, 200.0),
            scaling_record(1, 100.0),
            scaling_record(2, 150.0),
        ]

        rows = scaling_table(records)

        self.assertEqual([row["world_size"] for row in rows], [1, 2, 4])
        self.assertEqual([row["speedup"] for row in rows], [1.0, 1.5, 2.0])
        self.assertEqual([row["efficiency"] for row in rows], [1.0, 0.75, 0.5])
        self.assertEqual([row["status"] for row in rows], ["ok", "ok", "ok"])

    def test_scaling_table_needs_single_gpu_run(self):
        with self.assertRaisesRegex(ValueError, "one GPU"):
            scaling_table([scaling_record(2, 150.0)])
        with self.assertRaisesRegex(ValueError, "one GPU"):
            scaling_table([scaling_record(1, None, error="zu wenig Speicher")])

    def test_failed_run_stays_in_scaling_table(self):
        records = [scaling_record(1, 100.0), scaling_record(2, None, error="NaN")]

        rows = scaling_table(records)

        self.assertEqual(rows[1]["status"], "NaN")
        self.assertIsNone(rows[1]["speedup"])

    def test_two_runs_with_same_gpu_count_name_both_files(self):
        first, second = scaling_record(1, 100.0), scaling_record(1, 110.0)
        second["file"] = "noch-einmal.json"

        with self.assertRaisesRegex(ValueError, "scaling-gpus1.json.*noch-einmal.json"):
            scaling_table([first, second])


class SearchTableTest(TestCase):
    def test_search_table_marks_failed_and_missing(self):
        records = [
            search_record(1e-4, 8, loss=5.0),
            search_record(1e-4, 32, loss=5.5),
            search_record(1e-3, 8, error="Loss ist NaN"),
        ]

        rows = search_table(records, [1e-4, 1e-3], [8, 32])

        self.assertEqual(
            [(row["learning_rate"], row["batch_size"]) for row in rows],
            [(1e-4, 8), (1e-4, 32), (1e-3, 8), (1e-3, 32)],
        )
        self.assertEqual(
            [row["status"] for row in rows], ["ok", "ok", "Loss ist NaN", "fehlt"]
        )
        self.assertEqual([row["best_val_loss"] for row in rows], [5.0, 5.5, None, None])

    def test_duplicate_combination_names_both_files(self):
        first, second = (
            search_record(1e-4, 8, loss=5.0),
            search_record(1e-4, 8, loss=4.9),
        )
        second["file"] = "wiederholt.json"

        with self.assertRaises(ValueError) as caught:
            search_table([first, second], [1e-4], [8])
        self.assertIn("search-lr0.0001-bs8.json", str(caught.exception))
        self.assertIn("wiederholt.json", str(caught.exception))

    def test_record_outside_grid_is_an_error(self):
        # Etwa ein Lauf aus einem älteren Gitter im selben Ordner
        with self.assertRaisesRegex(ValueError, re.escape("search-lr0.01-bs8.json")):
            search_table([search_record(1e-2, 8, loss=6.0)], [1e-4], [8])

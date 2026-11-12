import hashlib
import tempfile
from pathlib import Path
from unittest import TestCase

from infm_sota2.model import GPT2_SMALL_CONFIG, download_weights, sha256_of


class Gpt2SmallConfigTest(TestCase):
    def test_matches_openai_gpt2_small(self):
        self.assertEqual(
            GPT2_SMALL_CONFIG,
            {
                "vocab_size": 50257,
                "context_length": 1024,
                "emb_dim": 768,
                "n_heads": 12,
                "n_layers": 12,
                "drop_rate": 0.1,
                "qkv_bias": True,
            },
        )


class DownloadWeightsTest(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.source = self.dir / "source.pth"
        self.source.write_bytes(b"pretend these are weights")
        self.sha256 = hashlib.sha256(b"pretend these are weights").hexdigest()
        self.dest = self.dir / "weights" / "gpt2.pth"

    def tearDown(self):
        self.tmp.cleanup()

    def test_sha256_of_matches_hashlib(self):
        self.assertEqual(sha256_of(self.source), self.sha256)

    def test_download_copies_file_with_matching_hash(self):
        path = download_weights(self.source.as_uri(), self.sha256, self.dest)

        self.assertEqual(path, self.dest)
        self.assertEqual(self.dest.read_bytes(), b"pretend these are weights")
        self.assertFalse(self.dest.with_name("gpt2.pth.part").exists())

    def test_download_rejects_wrong_hash_and_leaves_no_file(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            download_weights(self.source.as_uri(), "0" * 64, self.dest)

        self.assertFalse(self.dest.exists())
        self.assertFalse(self.dest.with_name("gpt2.pth.part").exists())

    def test_existing_file_is_verified_not_downloaded(self):
        self.dest.parent.mkdir()
        self.dest.write_bytes(b"pretend these are weights")
        missing = (self.dir / "does-not-exist.pth").as_uri()

        path = download_weights(missing, self.sha256, self.dest)

        self.assertEqual(path, self.dest)

    def test_existing_file_with_wrong_hash_raises(self):
        self.dest.parent.mkdir()
        self.dest.write_bytes(b"truncated")

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            download_weights(self.source.as_uri(), self.sha256, self.dest)

    def test_leftover_partial_download_is_replaced(self):
        self.dest.parent.mkdir()
        self.dest.with_name("gpt2.pth.part").write_bytes(b"half a fi")

        download_weights(self.source.as_uri(), self.sha256, self.dest)

        self.assertEqual(self.dest.read_bytes(), b"pretend these are weights")

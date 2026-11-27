import hashlib
import tempfile
from pathlib import Path
from unittest import TestCase

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from infm_sota2.model import (
    GPT2_LARGE_CONFIG,
    GPT2_MEDIUM_CONFIG,
    GPT2_SMALL_CONFIG,
    FeedForward,
    GPTModel,
    GPTModelV2,
    MultiHeadAttention,
    build_model,
    convert_v1_state_dict,
    download_weights,
    load_pretrained,
    sha256_of,
)
from tests.helpers import TINY_CONFIG, tiny_model


class GPTModelTest(TestCase):
    def test_gpt_model_returns_logits_per_token(self):
        model = tiny_model(seed=123)

        logits = model(torch.randint(0, 20, (3, 5)))

        self.assertEqual(logits.shape, (3, 5, 20))

    def test_logits_ignore_future_tokens(self):
        model = tiny_model(seed=123).eval()
        tokens = torch.randint(0, 20, (1, 6))
        changed = tokens.clone()
        changed[0, 4:] = (changed[0, 4:] + 1) % 20

        self.assertTrue(torch.allclose(model(tokens)[:, :4], model(changed)[:, :4]))


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


def v1_with_tanh_gelu(**overrides: int | float | bool):
    """V1 mit derselben GELU-Näherung wie V2, sonst weichen die Logits leicht ab."""
    model = tiny_model(seed=1, **overrides).eval()
    for module in model.modules():
        if isinstance(module, FeedForward):
            module.layers[1] = nn.GELU(approximate="tanh")
    return model


class GPTModelV2Test(TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tokens = torch.randint(0, 20, (2, 6))

    def converted(self, v1, **overrides: int | float | bool) -> GPTModelV2:
        v2 = GPTModelV2({**TINY_CONFIG, **overrides}).eval()
        v2.load_state_dict(convert_v1_state_dict(v1.state_dict()))
        return v2

    def test_matches_v1_with_converted_weights(self):
        v1 = v1_with_tanh_gelu()

        v2 = self.converted(v1)

        self.assertTrue(torch.allclose(v1(self.tokens), v2(self.tokens), atol=1e-5))

    def test_ignores_future_tokens(self):
        model = GPTModelV2(TINY_CONFIG).eval()
        changed = self.tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 20

        self.assertTrue(
            torch.allclose(model(self.tokens)[:, :4], model(changed)[:, :4])
        )

    def test_converts_without_qkv_bias(self):
        v1 = v1_with_tanh_gelu(qkv_bias=False)

        v2 = self.converted(v1, qkv_bias=False)

        self.assertTrue(torch.allclose(v1(self.tokens), v2(self.tokens), atol=1e-5))

    def test_math_kernel_gives_same_logits(self):
        model = GPTModelV2(TINY_CONFIG).eval()

        with sdpa_kernel(SDPBackend.MATH):
            math_logits = model(self.tokens)

        self.assertTrue(torch.allclose(model(self.tokens), math_logits, atol=1e-6))

    def test_has_no_mask_buffer(self):
        keys = GPTModelV2(TINY_CONFIG).state_dict()

        self.assertEqual([key for key in keys if "mask" in key], [])

    def test_load_pretrained_converts_v1_file(self):
        v1 = v1_with_tanh_gelu()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v1.pth"
            torch.save(v1.state_dict(), path)

            v2 = load_pretrained(GPTModelV2, TINY_CONFIG, path, CPU).eval()

        self.assertTrue(torch.allclose(v1(self.tokens), v2(self.tokens), atol=1e-5))

    def test_rejects_emb_dim_not_divisible_by_heads(self):
        with self.assertRaisesRegex(ValueError, "n_heads"):
            GPTModelV2({**TINY_CONFIG, "n_heads": 3})


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

    def test_medium_and_large_differ_from_small_in_size_only(self):
        # GPT-2 Medium und Large aus dem GPT-2-Paper (Tabelle 2)
        sizes = {"medium": (GPT2_MEDIUM_CONFIG, 24, 1024, 16)}
        sizes["large"] = (GPT2_LARGE_CONFIG, 36, 1280, 20)
        for name, (config, n_layers, emb_dim, n_heads) in sizes.items():
            with self.subTest(model=name):
                self.assertEqual(
                    config,
                    {
                        **GPT2_SMALL_CONFIG,
                        "n_layers": n_layers,
                        "emb_dim": emb_dim,
                        "n_heads": n_heads,
                    },
                )

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


def split_heads(x, num_heads):
    batch, num_tokens, d_out = x.shape
    return x.view(batch, num_tokens, num_heads, d_out // num_heads).transpose(1, 2)


class MultiHeadAttentionTest(TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.attention = MultiHeadAttention(
            d_in=3, d_out=4, context_length=6, dropout=0.0, num_heads=2
        )

    def test_returns_d_out_features_per_token(self):
        output = self.attention(torch.rand(2, 6, 3))

        self.assertEqual(output.shape, (2, 6, 4))

    def test_ignores_future_tokens(self):
        inputs = torch.rand(2, 6, 3)
        changed = inputs.clone()
        changed[:, 3:] = torch.rand(2, 3, 3)

        output = self.attention(inputs)
        changed_output = self.attention(changed)

        self.assertTrue(torch.allclose(output[:, :3], changed_output[:, :3]))

    def test_matches_pytorch_causal_attention(self):
        inputs = torch.rand(2, 6, 3)

        heads = F.scaled_dot_product_attention(
            split_heads(self.attention.W_query(inputs), num_heads=2),
            split_heads(self.attention.W_key(inputs), num_heads=2),
            split_heads(self.attention.W_value(inputs), num_heads=2),
            is_causal=True,
        )
        expected = self.attention.out_proj(heads.transpose(1, 2).reshape(2, 6, 4))

        self.assertTrue(torch.allclose(self.attention(inputs), expected, atol=1e-6))

    def test_rejects_d_out_not_divisible_by_num_heads(self):
        with self.assertRaisesRegex(
            AssertionError, "d_out must be divisible by num_heads"
        ):
            MultiHeadAttention(
                d_in=3, d_out=5, context_length=6, dropout=0.0, num_heads=2
            )

    def test_rejects_sequences_longer_than_context_length(self):
        with self.assertRaisesRegex(
            AssertionError, "Number of tokens 7 exceeds context length 6"
        ):
            self.attention(torch.rand(1, 7, 3))

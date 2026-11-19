import tempfile
from pathlib import Path
from unittest import TestCase

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from infm_sota2.model import GPTModelV2, convert_v1_state_dict, load_pretrained
from infm_sota2.transformer import FeedForward
from tests.helpers import TINY_CONFIG, tiny_model

CPU = torch.device("cpu")


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

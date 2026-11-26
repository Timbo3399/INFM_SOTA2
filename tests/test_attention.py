from unittest import TestCase

import torch
import torch.nn.functional as F

from infm_sota2.transformer import MultiHeadAttention


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

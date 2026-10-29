from unittest import TestCase

import torch
import torch.nn.functional as F

from infm_sota2 import MultiHeadAttention, create_dataloader
from infm_sota2.transformer import CausalAttention, SelfAttention


def split_heads(x, num_heads):
    batch, num_tokens, d_out = x.shape
    return x.view(batch, num_tokens, num_heads, d_out // num_heads).transpose(1, 2)


class SelfAttentionTest(TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.attention = SelfAttention(d_in=3, d_out=2)

    def test_supports_single_sequence(self):
        output = self.attention(torch.rand(6, 3))

        self.assertEqual(output.shape, (6, 2))

    def test_matches_pytorch_attention(self):
        inputs = torch.rand(6, 3)

        expected = F.scaled_dot_product_attention(
            self.attention.W_query(inputs).unsqueeze(0),
            self.attention.W_key(inputs).unsqueeze(0),
            self.attention.W_value(inputs).unsqueeze(0),
        ).squeeze(0)

        self.assertTrue(torch.allclose(self.attention(inputs), expected, atol=1e-6))


class CausalAttentionTest(TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.attention = CausalAttention(d_in=3, d_out=4, context_length=6, dropout=0.0)

    def test_returns_one_context_vector_per_token(self):
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

        expected = F.scaled_dot_product_attention(
            self.attention.W_query(inputs),
            self.attention.W_key(inputs),
            self.attention.W_value(inputs),
            is_causal=True,
        )

        self.assertTrue(torch.allclose(self.attention(inputs), expected, atol=1e-6))

    def test_supports_sequences_shorter_than_context_length(self):
        output = self.attention(torch.rand(2, 4, 3))

        self.assertEqual(output.shape, (2, 4, 4))

    def test_rejects_sequences_longer_than_context_length(self):
        with self.assertRaisesRegex(
            AssertionError, "Number of tokens 7 exceeds context length 6"
        ):
            self.attention(torch.rand(1, 7, 3))


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

    def test_processes_embedded_batches_from_dataloader(self):
        dataloader, embeddings, _ = create_dataloader(
            "hello world " * 20,
            seq_len=8,
            batch_size=2,
            emb_dim=4,
            shuffle=False,
        )
        attention = MultiHeadAttention(
            d_in=4, d_out=4, context_length=8, dropout=0.0, num_heads=2
        )

        inputs, _ = next(iter(dataloader))
        output = attention(embeddings(inputs))

        self.assertEqual(output.shape, (2, 8, 4))

from unittest import TestCase

import tiktoken
import torch
from torch.utils.data import DataLoader

from infm_sota2 import create_dataloader
from infm_sota2.data import BPEDataset, Embeddings


class CreateDataloaderTest(TestCase):
    def test_is_public_and_builds_components(self):
        dataloader, embeddings, dataset = create_dataloader(
            "hello world " * 20,
            seq_len=8,
            batch_size=2,
            shuffle=False,
        )

        self.assertIsInstance(dataloader, DataLoader)
        self.assertIsInstance(embeddings, Embeddings)
        self.assertIsInstance(dataset, BPEDataset)
        self.assertGreater(len(dataset), 0)

        inputs, targets = next(iter(dataloader))
        self.assertEqual(inputs.shape, targets.shape)
        self.assertEqual(inputs.shape[1], 8)
        self.assertEqual(inputs.dtype, torch.int64)
        self.assertEqual(targets.dtype, torch.int64)


class EmbeddingsTest(TestCase):
    def test_returns_token_and_position_embeddings(self):
        embeddings = Embeddings(vocab_size=16, emb_dim=4, max_len=3)

        output = embeddings(torch.zeros((2, 3), dtype=torch.long))

        self.assertEqual(output.shape, (2, 3, 4))

    def test_rejects_sequences_longer_than_max_len(self):
        embeddings = Embeddings(vocab_size=16, emb_dim=4, max_len=3)

        with self.assertRaisesRegex(
            AssertionError, "Sequence length 4 exceeds maximum length 3"
        ):
            embeddings(torch.zeros((1, 4), dtype=torch.long))


class BPEDatasetTest(TestCase):
    def test_decodes_tokens_and_exposes_vocabulary_size(self):
        dataset = BPEDataset("hello world " * 20, seq_len=8)

        inputs, _ = dataset[0]

        self.assertEqual(
            dataset.decode(inputs.tolist()),
            "hello world hello world hello world hello world",
        )
        self.assertGreater(dataset.vocab_size, 0)

    def test_samples_have_seq_len_tokens_and_shifted_targets(self):
        dataset = BPEDataset("hello world " * 20, seq_len=8)

        inputs, targets = dataset[0]

        self.assertEqual(inputs.shape, (8,))
        self.assertEqual(targets.shape, (8,))
        self.assertTrue(torch.equal(inputs[1:], targets[:-1]))
        self.assertIsInstance(dataset.tokens, torch.Tensor)

    def test_defaults_to_non_overlapping_windows(self):
        dataset = BPEDataset("hello world " * 20, seq_len=8)

        self.assertEqual(dataset.stride, 8)

    def test_rejects_text_shorter_than_sequence_length(self):
        with self.assertRaisesRegex(ValueError, "at least 9 tokens"):
            BPEDataset("hello", seq_len=8)

    def test_rejects_non_positive_stride(self):
        with self.assertRaisesRegex(ValueError, "stride must be positive"):
            BPEDataset("hello world " * 20, seq_len=8, stride=0)

    def test_rejects_non_positive_sequence_length(self):
        with self.assertRaisesRegex(ValueError, "seq_len must be positive"):
            BPEDataset("hello world " * 20, seq_len=0)

    def test_supports_overlapping_windows(self):
        dataset = BPEDataset("hello world " * 20, seq_len=8, stride=2)

        first_inputs, _ = dataset[0]
        second_inputs, _ = dataset[1]

        self.assertTrue(torch.equal(first_inputs[2:], second_inputs[:6]))

    def test_supports_exactly_one_window(self):
        text = "hello world " * 20
        token_count = len(tiktoken.get_encoding("gpt2").encode(text))
        dataset = BPEDataset(text, seq_len=token_count - 1)

        self.assertEqual(len(dataset), 1)
        inputs, targets = dataset[0]
        self.assertEqual(inputs.shape, (token_count - 1,))
        self.assertEqual(targets.shape, (token_count - 1,))

    def test_supports_stride_larger_than_sequence_length(self):
        dataset = BPEDataset("hello world " * 20, seq_len=8, stride=16)

        self.assertEqual(dataset.stride, 16)
        self.assertGreaterEqual(len(dataset), 1)

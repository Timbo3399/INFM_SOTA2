from unittest import TestCase

import tiktoken
import torch

from infm_sota2.data import BPEDataset


class BPEDatasetTest(TestCase):
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

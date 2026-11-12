from unittest import TestCase

import tiktoken
import torch

from infm_sota2.training import complete, generate
from tests.helpers import tiny_model


def seeded(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class GenerateTest(TestCase):
    def setUp(self):
        self.model = tiny_model(seed=123)
        self.prompt = torch.tensor([[1, 2, 3]])

    def run_generate(self, **kwargs):
        args = {
            "idx": self.prompt,
            "max_new_tokens": 5,
            "context_size": 8,
            "temperature": 0.0,
            "top_k": None,
            "generator": None,
        }
        return generate(self.model, **{**args, **kwargs})

    def test_appends_max_new_tokens(self):
        output = self.run_generate()

        self.assertEqual(output.shape, (1, 8))
        self.assertTrue(torch.equal(output[:, :3], self.prompt))

    def test_greedy_picks_argmax(self):
        output = self.run_generate(max_new_tokens=1)

        with torch.no_grad():
            expected = self.model.eval()(self.prompt)[0, -1].argmax()
        self.assertEqual(output[0, -1], expected)

    def test_top_k_one_equals_greedy(self):
        greedy = self.run_generate()
        sampled = self.run_generate(temperature=1.0, top_k=1, generator=seeded(7))

        self.assertTrue(torch.equal(greedy, sampled))

    def test_same_seed_gives_same_sample(self):
        first = self.run_generate(temperature=1.0, top_k=5, generator=seeded(7))
        second = self.run_generate(temperature=1.0, top_k=5, generator=seeded(7))

        self.assertTrue(torch.equal(first, second))

    def test_crops_prompt_longer_than_context(self):
        long_prompt = torch.arange(10).unsqueeze(0) % 20

        output = self.run_generate(idx=long_prompt, max_new_tokens=1)

        with torch.no_grad():
            expected = self.model.eval()(long_prompt[:, -8:])[0, -1].argmax()
        self.assertEqual(output.shape, (1, 11))
        self.assertEqual(output[0, -1], expected)

    def test_rejects_negative_temperature(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            self.run_generate(temperature=-0.5)

    def test_rejects_top_k_below_one(self):
        with self.assertRaisesRegex(ValueError, "top_k"):
            self.run_generate(temperature=1.0, top_k=0, generator=seeded(7))

    def test_restores_training_mode(self):
        self.model.train()

        self.run_generate()

        self.assertTrue(self.model.training)


class CompleteTest(TestCase):
    def test_complete_returns_only_continuation(self):
        tokenizer = tiktoken.get_encoding("gpt2")
        model = tiny_model(seed=123, vocab_size=tokenizer.n_vocab, emb_dim=8)
        prompt_ids = tokenizer.encode("The Elves are")

        text = complete(
            model,
            tokenizer,
            "The Elves are",
            max_new_tokens=4,
            context_size=8,
            temperature=0.0,
            top_k=None,
            generator=None,
        )

        full = generate(
            model,
            torch.tensor([prompt_ids]),
            max_new_tokens=4,
            context_size=8,
            temperature=0.0,
            top_k=None,
            generator=None,
        )
        self.assertEqual(text, tokenizer.decode(full[0, len(prompt_ids) :].tolist()))

from unittest import TestCase

import torch

from tests.helpers import tiny_model


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

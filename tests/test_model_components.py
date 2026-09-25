"""
Model tests that would have caught the v1 defects:
  - length regulator dropping short tokens (defect 1)
  - flow-matching sampler integrating the wrong way (defect 3)
  - duration predictor LayerNorm crash, unmasked encoder convs, untrained aligner (found while fixing)
"""
import itertools
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.model.duration import (DurationPredictor, durations_from_logw, gaussian_log_prior,
                                maximum_path, path_from_durations)
from src.model.flow_matching import CFMDecoder, repeat_skip_negatives, spfm_route
from src.model.text_encoder import TextEncoder, sequence_mask

torch.manual_seed(0)


class TestTextEncoder:
    def make(self):
        return TextEncoder(n_vocab=60, n_mels=20, dim=64, n_heads=2, ff_dim=128, n_layers=2, dropout=0.0).eval()

    def test_shapes(self):
        enc = self.make()
        h, mu, mask = enc(torch.randint(1, 60, (2, 17)), torch.zeros(2, 17, dtype=torch.long), torch.tensor([17, 9]))
        assert h.shape == (2, 64, 17) and mu.shape == (2, 20, 17) and mask.shape == (2, 1, 17)
        assert (mu[1, :, 9:] == 0).all()

    def test_padding_invariance_is_exact(self):
        """v1 leaked padding through unmasked convs (its test tolerated a 0.5 mean difference)."""
        enc = self.make()
        ids = torch.randint(1, 60, (1, 12))
        langs = torch.zeros(1, 12, dtype=torch.long)
        h1, mu1, _ = enc(ids, langs, torch.tensor([12]))
        ids_p = torch.cat([ids, torch.zeros(1, 20, dtype=torch.long)], dim=1)
        h2, mu2, _ = enc(ids_p, torch.zeros(1, 32, dtype=torch.long), torch.tensor([12]))
        assert torch.allclose(mu1, mu2[:, :, :12], atol=1e-5)

    def test_language_embedding_changes_encoding(self):
        enc = self.make()
        ids = torch.randint(1, 60, (1, 8))
        a, _, _ = enc(ids, torch.zeros(1, 8, dtype=torch.long), torch.tensor([8]))
        b, _, _ = enc(ids, torch.full((1, 8), 2, dtype=torch.long), torch.tensor([8]))
        assert not torch.allclose(a, b)


class TestDurationPredictor:
    def test_any_length(self):
        """v1: nn.LayerNorm(hidden) on (B, hidden, T) crashed whenever T != hidden."""
        dp = DurationPredictor(in_dim=64, hidden=32)
        for T in (1, 7, 31, 200):
            out = dp(torch.randn(2, 64, T), torch.ones(2, 1, T))
            assert out.shape == (2, 1, T)

    def test_masked_positions_are_zero(self):
        dp = DurationPredictor(in_dim=64, hidden=32)
        mask = sequence_mask(torch.tensor([10, 4]), 10)
        out = dp(torch.randn(2, 64, 10), mask)
        assert (out[1, :, 4:] == 0).all()

    def test_encoder_input_is_detached(self):
        dp = DurationPredictor(in_dim=16, hidden=16)
        h = torch.randn(1, 16, 5, requires_grad=True)
        dp(h, torch.ones(1, 1, 5)).sum().backward()
        assert h.grad is None

    def test_tempo_conditioning_moves_durations(self):
        dp = DurationPredictor(in_dim=16, hidden=16, cond_dim=4)
        for film in dp.films:
            nn.init.normal_(film.weight, std=0.5)
        h = torch.randn(1, 16, 6)
        a = dp(h, torch.ones(1, 1, 6), torch.tensor([[1.0, 0, 0, 0]]))
        b = dp(h, torch.ones(1, 1, 6), torch.tensor([[-1.0, 0, 0, 0]]))
        assert not torch.allclose(a, b)


def brute_force_best(value: np.ndarray):
    tx, ty = value.shape
    best, best_path = -np.inf, None
    for cuts in itertools.combinations(range(1, ty), tx - 1):  # token boundaries
        bounds = (0, *cuts, ty)
        score = sum(value[i, bounds[i]:bounds[i + 1]].sum() for i in range(tx))
        if score > best:
            best, best_path = score, bounds
    return best, best_path


class TestMAS:
    @pytest.mark.parametrize("tx,ty,seed", [(1, 5, 0), (3, 3, 1), (3, 7, 2), (4, 9, 3), (5, 8, 4)])
    def test_matches_exhaustive_search(self, tx, ty, seed):
        rng = np.random.default_rng(seed)
        value = rng.standard_normal((tx, ty)).astype(np.float32)
        path = maximum_path(torch.from_numpy(value)[None], torch.tensor([tx]), torch.tensor([ty]))[0].numpy()
        best, _ = brute_force_best(value)
        assert math.isclose(float((path * value).sum()), best, rel_tol=1e-5, abs_tol=1e-5)
        assert (path.sum(0) == 1).all()          # every frame -> exactly one token
        assert (path.sum(1) >= 1).all()          # every token >= 1 frame
        firsts = path.argmax(1)
        assert (np.diff(firsts) > 0).all()       # monotonic

    def test_rejects_fewer_frames_than_tokens(self):
        with pytest.raises(ValueError):
            maximum_path(torch.randn(1, 5, 3), torch.tensor([5]), torch.tensor([3]))

    def test_log_prior_matches_direct_gaussian(self):
        mu, y = torch.randn(1, 4, 3), torch.randn(1, 4, 5)
        lp = gaussian_log_prior(mu, y)
        direct = torch.distributions.Normal(mu.transpose(1, 2).unsqueeze(2), 1.0).log_prob(
            y.transpose(1, 2).unsqueeze(1)).sum(-1)
        assert torch.allclose(lp, direct, atol=1e-4)


class TestLengthRegulation:
    def test_no_token_is_dropped(self):
        """v1: round() + dur > 0 filter made 0.49, 0.5 and tiny predictions vanish."""
        frames = torch.tensor([[3.2, 0.49, 0.5, 0.51, 1.5, 2.5, 0.01]])
        logw = torch.log(frames).unsqueeze(1)
        mask = torch.ones(1, 1, 7)
        for scale in (1.0, 0.5, 0.25, 1.7):
            d = durations_from_logw(logw, mask, scale)
            assert (d >= 1).all(), (scale, d)

    def test_padding_tokens_get_zero_frames(self):
        mask = sequence_mask(torch.tensor([3]), 5)
        d = durations_from_logw(torch.zeros(1, 1, 5), mask)
        assert d[0, 0].tolist() == [1, 1, 1, 0, 0]

    def test_path_is_monotonic_and_complete(self):
        d = torch.tensor([[[2.0, 1.0, 3.0, 0.0]]])
        mask = torch.tensor([[[1.0, 1.0, 1.0, 0.0]]])
        p = path_from_durations(d, mask)[0]
        assert p.shape == (4, 6)
        assert p.sum(1).tolist() == [2, 1, 3, 0] and (p.sum(0) == 1).all()
        assert p[0, :2].all() and p[1, 2] == 1 and p[2, 3:].all()


class GaussianOracle(nn.Module):
    """Exact optimal velocity for 1-D data N(m, s^2) under the Matcha convention."""

    def __init__(self, m, s, sigma_min):
        super().__init__()
        self.m, self.s, self.k = m, s, 1 - sigma_min

    def forward(self, x, mask, mu, t, cond=None):
        t = t[:, None, None]
        a = 1 - self.k * t
        var = a ** 2 + (t * self.s) ** 2
        return self.m + (t * self.s ** 2 - self.k * a) / var * (x - t * self.m)


class TestFlowMatchingDirection:
    def test_sampler_transports_noise_to_data(self):
        """With the exact optimal field, sampling must recover the data distribution.
        v1's sampler, fed the same kind of field, produced N(+2.5, 0.5^2) from N(-5, 2^2) data."""
        dec = CFMDecoder(n_mels=1, dim=8, levels=1)
        dec.estimator = GaussianOracle(-5.0, 2.0, dec.sigma_min)
        torch.manual_seed(0)
        x = dec.sample(torch.zeros(1, 1, 20000), torch.ones(1, 1, 20000), n_steps=200, temperature=1.0)
        assert abs(x.mean().item() + 5.0) < 0.1 and abs(x.std().item() - 2.0) < 0.1

    def test_training_target_matches_sampler(self):
        """The field the sampler needs must be the minimiser of the training loss (same convention)."""
        dec = CFMDecoder(n_mels=1, dim=8, levels=1)
        x1 = -5.0 + 2.0 * torch.randn(1, 1, 50000)
        mask = torch.ones(1, 1, 50000)

        class Shifted(nn.Module):
            def __init__(self, base, delta):
                super().__init__(); self.base, self.delta = base, delta

            def forward(self, *a):
                return self.base(*a) + self.delta

        def loss_with(field):
            dec.estimator = field
            torch.manual_seed(1)
            return dec.loss(x1, mask, torch.zeros_like(x1)).item()

        oracle = GaussianOracle(-5.0, 2.0, dec.sigma_min)
        l_oracle = loss_with(oracle)
        assert l_oracle < loss_with(GaussianOracle(+5.0, 2.0, dec.sigma_min))  # mirrored field
        assert l_oracle < loss_with(Shifted(oracle, 0.3)) and l_oracle < loss_with(Shifted(oracle, -0.3))

    def test_overfits_one_utterance(self):
        """A tiny decoder must learn to reconstruct a fixed mel from its prior: catches any sign/direction bug."""
        torch.manual_seed(0)
        n_mels, T = 16, 32
        target = torch.sin(torch.linspace(0, 6, T))[None, None, :] * torch.linspace(0.5, 1.5, n_mels)[None, :, None]
        mu = target + 0.3 * torch.randn_like(target)  # a noisy prior, like an early encoder
        mask = torch.ones(1, 1, T)
        dec = CFMDecoder(n_mels, dim=64, levels=2)
        opt = torch.optim.Adam(dec.parameters(), lr=2e-3)
        for _ in range(400):
            opt.zero_grad()
            dec.loss(target.expand(8, -1, -1), mask.expand(8, -1, -1), mu.expand(8, -1, -1)).backward()
            opt.step()
        dec.eval()
        out = dec.sample(mu, mask, n_steps=16, temperature=0.5)
        err = ((out - target) ** 2).mean().item()
        assert err < 0.05 * target.var().item() + 0.02, err


class TestRobustnessHelpers:
    def test_negatives_preserve_length_and_differ(self):
        x = torch.randn(4, 10, 40)
        neg = repeat_skip_negatives(x, torch.tensor([40, 40, 30, 30]))
        assert neg.shape == x.shape
        assert all(not torch.allclose(neg[b], x[b]) for b in range(4))

    def test_spfm_routing(self):
        route = spfm_route(torch.tensor([0.2, 0.9]), torch.tensor([0.5, 0.6]))
        assert route.tolist() == [False, True]

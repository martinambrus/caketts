"""Tests for the multilingual SSL discriminator wrapper, tempo labels, pace metrics and ASR checks."""
import pytest
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

from src.data.tempo import flag_rate_outliers, tempo_from_alignment
from src.eval.asr_check import check_sentence, retry_actions
from src.eval.pace import Sentence, m1_sentence_outliers, m2_end_slowdown, m4_cross_text_cv, m6_distribution_distance
from src.model.discriminators import SSLDiscriminator
from src.model.duration import durations_from_logw


@pytest.fixture(scope="module")
def tiny_ssl():
    # random-initialised stand-in with the XLS-R architecture (no download in tests)
    cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                         conv_dim=(32, 32), conv_kernel=(10, 3), conv_stride=(5, 2), num_feat_extract_layers=2,
                         feat_extract_norm="layer", do_stable_layer_norm=True)
    return Wav2Vec2Model(cfg)


class TestSSLDiscriminator:
    def test_encoder_frozen_and_generator_gets_gradients(self, tiny_ssl):
        d = SSLDiscriminator(tiny_ssl, n_layers=3, hidden=32, head_dim=16)
        real = torch.randn(2, 24000) * 0.1
        fake = (torch.randn(2, 24000) * 0.1).requires_grad_(True)
        adv, fm = d.generator_loss(real, fake)
        (adv + fm).backward()
        assert fake.grad is not None and fake.grad.abs().sum() > 0      # 24k -> 16k resample is differentiable
        assert all(p.grad is None for p in d.ssl.parameters())           # frozen backbone

    def test_discriminator_step_updates_head_only(self, tiny_ssl):
        d = SSLDiscriminator(tiny_ssl, n_layers=3, hidden=32, head_dim=16).train()
        assert not d.ssl.training                                        # stays in eval mode
        loss = d.discriminator_loss(torch.randn(2, 12000), torch.randn(2, 12000))
        loss.backward()
        assert any(p.grad is not None for p in d.head.parameters())


class TestTempo:
    def test_articulation_rate_excludes_pauses(self):
        kinds = ["phone"] * 10 + ["punct"] + ["phone"] * 10
        frames = [5] * 10 + [47] + [5] * 10   # 47 frames * 10.67 ms = 0.5 s pause
        lab = tempo_from_alignment(kinds, frames, hop_s=256 / 24000)
        speech = 100 * 256 / 24000
        assert abs(lab.articulation_rate - 20 / speech) < 1e-6
        assert 0.3 < lab.pause_ratio < 0.4

    def test_outlier_flags(self):
        rates = {f"u{i}": 12.0 for i in range(20)} | {"slow": 9.5}
        assert flag_rate_outliers(rates) == ["slow"]


class TestPace:
    def test_m2_catches_end_of_text_slowdown(self):
        s = [Sentence(60, 5.0, i, 8) for i in range(6)] + [Sentence(60, 5.9, 6, 8), Sentence(60, 5.9, 7, 8)]
        m = m2_end_slowdown(s)
        assert m["ratio"] < 0.95 and m["slope"] < 0                       # ElevenLabs-style wind-down fails
        flat = [Sentence(60, 5.0, i, 8) for i in range(8)]
        assert abs(m2_end_slowdown(flat)["ratio"] - 1.0) < 1e-9

    def test_m1_m4_m6(self):
        s = [Sentence(60, 5.0, 0, 3), Sentence(60, 6.0, 1, 3), Sentence(60, 5.1, 2, 3)]
        assert m1_sentence_outliers(s, target_rate=12.0) == [1]
        assert m4_cross_text_cv([12.0, 12.0, 12.0]) == 0.0
        assert m6_distribution_distance([12, 12.5, 13], [12, 12.5, 13]) < 1e-9


class TestASRCheck:
    REF = "Včera večer sme s bratom išli do kina a potom domov pešo."

    def test_clean_render_passes(self):
        assert check_sentence(self.REF, "včera večer sme s bratom išli do kina a potom domov pešo", 0.03).ok

    def test_skip_is_caught_by_run_length(self):
        # three contiguous words missing: flagged as a skip even with a generous CER margin
        r = check_sentence(self.REF, "včera večer sme s bratom a potom domov pešo", 0.30)
        assert not r.ok and any(x.startswith("skip") for x in r.reasons)
        assert r.suspect_word_indices == [5, 6, 7]

    def test_cut_off_ending(self):
        r = check_sentence(self.REF, "včera večer sme s bratom išli do kina a potom", 0.03)
        assert not r.ok and any("ending" in x for x in r.reasons)

    def test_repetition_is_caught(self):
        r = check_sentence(self.REF, "včera večer sme s bratom išli do kina do kina a potom domov pešo", 0.03)
        assert not r.ok

    def test_retry_changes_something_that_matters(self):
        r = check_sentence(self.REF, "včera večer sme s bratom išli do kina a potom", 0.03)
        assert "raise local length_scale (x1.15) on the suspect words" in retry_actions(r)

    def test_token_scale_slows_only_suspect_tokens(self):
        logw = torch.log(torch.full((1, 1, 4), 3.0))
        scale = torch.tensor([[1.0, 1.0, 1.5, 1.0]])
        d = durations_from_logw(logw, torch.ones(1, 1, 4), 1.0, scale)
        assert d[0, 0].tolist() == [3, 3, 5, 3]

"""
End-to-end: train a tiny MatchaTTS on a synthetic 'language' whose tokens have known spectra and
durations, then check that MAS finds the true segmentation, the duration predictor learns it,
synthesis keeps every token, and the generated mel has the right content.
This is the test v1 lacked: shape tests passed while the model could never have worked.
"""
import pytest
import torch

from src.model.matcha_tts import MatchaConfig, MatchaTTS


def synthetic_batch(n_utts=16, n_mels=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    V = 10
    patterns = torch.randn(V, n_mels, generator=g) * 1.5
    true_dur = {k: 2 + (k % 4) for k in range(1, V)}          # 2..5 frames per token
    seqs, mels, durs = [], [], []
    for _ in range(n_utts):
        L = int(torch.randint(4, 9, (1,), generator=g))
        s = [int(torch.randint(1, V, (1,), generator=g))]
        while len(s) < L:  # no identical neighbours: their boundary would be unobservable
            k = int(torch.randint(1, V, (1,), generator=g))
            if k != s[-1]:
                s.append(k)
        s = torch.tensor(s)
        d = torch.tensor([true_dur[int(k)] for k in s])
        m = torch.cat([patterns[k].unsqueeze(1).repeat(1, true_dur[int(k)]) for k in s], dim=1)
        seqs.append(s); durs.append(d); mels.append(m + 0.05 * torch.randn(m.shape, generator=g))
    B = n_utts
    tx = torch.tensor([len(s) for s in seqs]); ty = torch.tensor([m.shape[1] for m in mels])
    ids = torch.zeros(B, int(tx.max()), dtype=torch.long); y = torch.zeros(B, n_mels, int(ty.max()))
    dd = torch.zeros(B, int(tx.max()))
    for i in range(B):
        ids[i, : tx[i]] = seqs[i]; y[i, :, : ty[i]] = mels[i]; dd[i, : tx[i]] = durs[i]
    batch = {"phoneme_ids": ids, "phoneme_lengths": tx, "mels": y, "mel_lengths": ty,
             "lang_ids": torch.zeros_like(ids), "speaker_ids": torch.zeros(B, dtype=torch.long),
             "boundary_ids": torch.zeros(B, dtype=torch.long), "rates": torch.tensor([[12.0, 0.0]]).repeat(B, 1)}
    return batch, dd, patterns


@pytest.mark.slow
def test_tiny_matcha_learns_alignment_durations_and_content():
    torch.manual_seed(0)
    batch, true_d, patterns = synthetic_batch()
    cfg = MatchaConfig(n_vocab=10, n_mels=16, enc_dim=64, enc_layers=2, enc_heads=2, enc_ff=128,
                       spk_dim=8, cond_dim=8, dur_hidden=64, dec_dim=64, dec_levels=2, dropout=0.0)
    model = MatchaTTS(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for step in range(700):
        opt.zero_grad()
        out = model(batch)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()

    # 1) MAS recovers the true segmentation
    with torch.no_grad():
        attn = model(batch)["attn"]
    mas_d = attn.sum(-1)
    valid = batch["phoneme_ids"] > 0
    assert ((mas_d - true_d).abs()[valid] <= 1).float().mean() > 0.9

    # 2) + 3) synthesis: every token kept, durations learned, content right
    res = model.synthesize(batch["phoneme_ids"], batch["lang_ids"], batch["phoneme_lengths"],
                           batch["speaker_ids"], batch["rates"], batch["boundary_ids"],
                           n_steps=10, temperature=0.3)
    pred_d = res["durations"]
    assert (pred_d[valid] >= 1).all()
    assert (pred_d - true_d).abs()[valid].mean() < 1.0
    cos = []
    for b in range(batch["phoneme_ids"].shape[0]):
        start = 0
        for i in range(int(batch["phoneme_lengths"][b])):
            n = int(pred_d[b, i])
            seg = res["mel"][b, :, start:start + n].mean(dim=1)
            ref = patterns[batch["phoneme_ids"][b, i]]
            cos.append(torch.nn.functional.cosine_similarity(seg, ref, dim=0))
            start += n
    assert torch.stack(cos).mean() > 0.9

# Recommended TTS Architecture for Czech Voice Cloning
## Optimized for 10+ Hours of Speaker Data, Maximum Quality, Minimal Hallucinations

> **Update, 25 September 2026.** This November 2025 rationale still holds for the core: Matcha-TTS with explicit durations and MAS, plus BigVGAN v2. Five statements below were wrong or overstated and are corrected in place:
> 1. **WavLM is English-only.** Its 94k pretraining hours are Libri-Light, GigaSpeech and English VoxPopuli. The SSL discriminator now uses XLS-R-300M, which has Czech and Slovak in its pretraining data, and is chosen by ablation.
> 2. **The espeak-ng example output was wrong,** and espeak-ng 1.52 does *not* handle Czech/Slovak voicing assimilation reliably. The G2P now has a lexicon, post-rules and a strict vocabulary.
> 3. **ECAPA-TDNN conditioning is replaced** by a learned narrator embedding. Per-utterance vectors carry session and language information.
> 4. **"0% hallucinations, guaranteed" was overstated.** Explicit durations minimise skips structurally; every rendered sentence is still verified by ASR.
> 5. **"WER 2–4%" was unmeasurable.** The best open Czech/Slovak ASR errs on 5.5–11% of words of human speech, so targets are set relative to the narrator's own recordings.
>
> The implementation plan v2 (`czech_slovak_tts_implementation_plan.md`) is authoritative. The research behind these changes is in *TTS breakthroughs since late 2025*.

---

## Executive Summary

**Recommended Base Architecture: Matcha-TTS + StyleTTS2 Components**

Given your requirements (Czech voice, 10+ hours of high-quality data, maximum naturalness, no hallucinations), I recommend building a **hybrid architecture** that combines:

1. **Matcha-TTS** as the acoustic backbone (NAR flow-matching, no ASR dependency, explicit durations that make skips and repeats structurally rare)
2. **StyleTTS2's SLM discriminator idea** (adversarial training through a frozen speech SSL model), with a **multilingual** backbone (XLS-R-300M) instead of English-only WavLM
3. **BigVGAN v2** as the vocoder (SOTA quality, 44kHz support)
4. **espeak-ng phonemizer** for Czech/Slovak G2P (native support, no training), with a lexicon and post-rules for its known errors (v2)

This combination gives you StyleTTS2's quality innovations without its ASR dependency problem, while keeping the structural robustness of non-autoregressive generation with explicit durations.

---

## Why This Architecture

### The Core Problem with StyleTTS2 for Czech

StyleTTS2 achieves human-level quality through three key innovations:
1. **Style diffusion** - models speech style as latent variable
2. **WavLM SLM discriminator** - adversarial training with speech language model
3. **Differentiable duration modeling** - end-to-end training with duration prediction

However, it requires:
- **ASR text aligner** - pre-trained only for EN/JA/ZH, requires training from scratch for Czech
- **PL-BERT** - only English pretrained, multilingual version exists but 14 languages (Czech not included)

The ASR aligner is the hardest blocker - it's used for computing ground truth alignments during training, and training a new one requires a separate large Czech ASR corpus.

### Why Matcha-TTS Solves This

Matcha-TTS uses **Monotonic Alignment Search (MAS)** to learn alignments directly from data - no external ASR needed. It:
- Learns to speak and align without external aligners
- Uses OT-CFM (Optimal Transport Conditional Flow Matching) for high-quality generation
- Is fully non-autoregressive with explicit durations - **no autoregressive drift**; skips become rare rather than impossible, so v2 still verifies every sentence
- Has the smallest memory footprint of comparable models
- Achieves highest MOS scores while being fast

### The Hybrid Approach

We can enhance Matcha-TTS with StyleTTS2's best components:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT TEXT (Czech)                           │
├─────────────────────────────────────────────────────────────────────┤
│  TEXT NORMALIZATION                                                 │
│  • num2words (lang='cs')                                            │
│  • Custom rules for Czech abbreviations, dates, numbers             │
├─────────────────────────────────────────────────────────────────────┤
│  G2P (Grapheme-to-Phoneme)                                         │
│  • phonemizer + espeak-ng (cs, sk, en-us) + lexicon + post-rules    │
│  • Strict vocabulary built from real espeak-ng output (v2)          │
├─────────────────────────────────────────────────────────────────────┤
│  TEXT ENCODER                                                       │
│  • Matcha-TTS encoder (Transformer-based)                          │
│  • Alternative: XPhoneBERT for multilingual phoneme embeddings     │
├─────────────────────────────────────────────────────────────────────┤
│  DURATION PREDICTOR + MAS ALIGNMENT                                │
│  • Matcha-TTS duration predictor                                   │
│  • Monotonic Alignment Search (learns alignment without ASR)        │
├─────────────────────────────────────────────────────────────────────┤
│  NARRATOR + TEMPO CONDITIONING (v2)                                 │
│  • Learned narrator embedding (speaker table), not ECAPA vectors    │
│  • Articulation-rate + pause inputs to the duration predictor,      │
│    fixed at inference; paragraph/chapter boundary flag              │
├─────────────────────────────────────────────────────────────────────┤
│  ACOUSTIC MODEL (Flow Matching Decoder)                            │
│  • Matcha-TTS U-Net decoder with OT-CFM                            │
│  • Outputs mel-spectrogram                                          │
├─────────────────────────────────────────────────────────────────────┤
│  DISCRIMINATORS (StyleTTS2 Enhancement)                            │
│  • SSL discriminator: XLS-R-300M (frozen) + trainable head (v2)    │
│  • Multi-Period Discriminator (MPD)                                 │
│  • Multi-Resolution Discriminator (MRD)                            │
├─────────────────────────────────────────────────────────────────────┤
│  VOCODER                                                            │
│  • BigVGAN v2 (24kHz or 44kHz)                                     │
│  • Snake activation + anti-aliased representation                   │
│  • Pre-trained, fine-tune on Czech if needed                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Analysis

### 1. Base: Matcha-TTS

**Architecture:**
- Encoder-decoder with 1D U-Net decoder
- Trained with OT-CFM (Optimal Transport Conditional Flow Matching)
- MAS for learning alignments without external tools
- ~18M parameters (acoustic model only)

**Key advantages:**
- Learns to speak from scratch without external alignments
- Probabilistic but non-autoregressive (no AR-style looping or drift)
- 2-4 synthesis steps sufficient for high quality (vs 10+ for diffusion)
- Smallest memory footprint in its class
- Clean codebase, easy to modify

**Training on Czech:**
```yaml
# Config for Czech Matcha-TTS
data:
  train_filelist: data/czech/train.txt
  valid_filelist: data/czech/val.txt
  text_cleaners: [phonemize_text]  # Custom cleaner using espeak-ng
  language: cs
  
model:
  encoder:
    n_vocab: 150  # IPA phoneme inventory
    n_layers: 6
    n_heads: 2
    
  decoder:
    dim: 256
    n_layers: 6
```

### 2. Enhancement: SSL Discriminator (multilingual backbone in v2)

This is StyleTTS2's key innovation for naturalness. The discriminator:

- Uses a frozen speech SSL encoder with a trainable discriminative head
- Captures semantic and acoustic aspects humans perceive
- **Correction (v2):** WavLM-Large was pretrained on English only (Libri-Light, GigaSpeech, English VoxPopuli), so v2 uses XLS-R-300M (436k h, 128 languages incl. cs/sk, Apache-2.0) and ablates it against Omnilingual wav2vec 2.0, w2v-BERT 2.0 and no SSL discriminator. The v1 sketch below also had the wrong dimensions (wavlm-large returns 25 hidden states of 1024 dims, not 13 × 768). Use `src/model/discriminators.py` from plan v2 Step 9.

**Implementation:**
```python
class SLMDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Frozen WavLM encoder
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        for param in self.wavlm.parameters():
            param.requires_grad = False
            
        # Trainable discriminator head
        self.head = nn.Sequential(
            nn.Linear(13 * 768, 256),  # All 13 layers concatenated
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, audio):
        # Downsample to 16kHz for WavLM
        audio_16k = torchaudio.transforms.Resample(24000, 16000)(audio)
        features = self.wavlm(audio_16k, output_hidden_states=True)
        # Concatenate all layer outputs
        all_layers = torch.cat(features.hidden_states, dim=-1)
        return self.head(all_layers.mean(dim=1))
```

**Why this needs a multilingual backbone for Czech (corrected):**
WavLM's 94k hours are English only, and SSL models typically degrade on unfamiliar languages. StyleTTS2's ablation (−0.32 CMOS without the SLM discriminator) was measured on English, so the gain for Czech/Slovak has to be measured by ablation.

### 3. Vocoder: BigVGAN v2

**Why BigVGAN v2:**
- Snake activation provides periodic inductive bias for speech
- Anti-aliased representation prevents high-frequency artifacts
- 112M parameters trained for 5M steps
- Multi-scale sub-band CQT discriminator (v2 improvement)
- CUDA kernel for 1.5-3x faster inference

**Configuration:**
```python
# Use 24kHz for balanced quality/speed, 44kHz for maximum quality
bigvgan_config = {
    "model": "nvidia/bigvgan_v2_24khz_100band_256x",  # or 44khz for max quality
    "use_cuda_kernel": True,  # 1.5-3x speedup on H200/B200
}
```

**Fine-tuning on Czech (optional):**
BigVGAN's zero-shot generalization is excellent, but fine-tuning on your narrator's voice can improve quality:
```bash
python train.py \
    --input_training_file czech_train.txt \
    --checkpoint_path nvidia/bigvgan_v2_24khz_100band_256x \
    --training_epochs 100 \
    --batch_size 32  # H200 can handle 64-128
```

### 4. G2P: espeak-ng Phonemizer

Czech is well-supported by espeak-ng:

```python
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend

# Czech phonemization
backend = EspeakBackend('cs', language_switch='remove-flags')
text = "Dobrý den, jak se máte?"
phonemes = phonemize(text, backend='espeak', language='cs')
# espeak-ng 1.52 actually outputs: "dˈobriː dˈen jˈak se mˈaːte"   (v1 showed an invented "dobrɪː dɛn ...")
```

**Czech-specific considerations:**
- Voice assimilation (voicing spreads across word boundaries)
- Final devoicing
- Syllabic consonants (r, l can be syllable nuclei)

**Correction (v2):** espeak-ng 1.52 handles syllabic consonants and final devoicing before a pause, but not voicing across word boundaries ("k domu" → k | d…, "muž je" → ʒ | j…). It devoices "zh" ("rozhlas" → [roshlas]) and splits Slovak ô. Plan v2 Step 3 adds a lexicon, word rules and cross-word voicing rules, each covered by tests.

---

## Architectural Enhancements from Other Systems

### From StyleTTS2: Style Diffusion (Optional)

If you want expressive/varied prosody (useful for audiobook narration), add style diffusion:

```python
class StyleDiffusion(nn.Module):
    """Sample style vectors for text-appropriate prosody"""
    def __init__(self, style_dim=128):
        super().__init__()
        self.denoiser = EDMDenoiser(style_dim)
        
    def sample(self, text_embedding, num_steps=3):
        # Start from noise
        style = torch.randn(text_embedding.shape[0], 128)
        # Denoise conditioned on text
        for t in range(num_steps):
            style = self.denoiser(style, text_embedding, t)
        return style
```

This allows sampling diverse but appropriate styles for each sentence without reference audio.

### From F5-TTS: ConvNeXt Text Refinement

F5-TTS's key insight is using ConvNeXt blocks to refine text representation before alignment:

```python
class ConvNeXtTextRefiner(nn.Module):
    """Refines text embeddings for better alignment"""
    def __init__(self, dim=256, depth=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(dim) for _ in range(depth)
        ])
        
    def forward(self, text_emb):
        for block in self.blocks:
            text_emb = block(text_emb)
        return text_emb
```

This addresses E2-TTS's "slow convergence and low robustness" issues.

### From BigVGAN: Snake Activation for Decoder

Consider replacing decoder activations with Snake:

```python
class Snake(nn.Module):
    """Periodic activation for audio synthesis"""
    def __init__(self, channels, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1) * alpha)
        
    def forward(self, x):
        return x + (1 / self.alpha) * torch.sin(self.alpha * x) ** 2
```

Snake provides periodic inductive bias that helps model speech harmonics.

---

## Training Pipeline

### Phase 1: Data Preparation

```bash
# 1. Audio preprocessing
# Normalize to -23 LUFS, remove silence, ensure 24kHz
python preprocess_audio.py --input_dir raw_audio/ --output_dir processed/

# 2. Generate phoneme transcriptions
python generate_phonemes.py --language cs --input transcripts.txt

# 3. Create training manifest
python create_manifest.py \
    --audio_dir processed/ \
    --phonemes phonemes.txt \
    --output train_manifest.txt
```

### Phase 2: Matcha-TTS Training

```bash
# Train Matcha-TTS acoustic model
python train.py \
    --config configs/czech_matcha.yaml \
    --gpus 8 \  # H200 cluster
    --batch_size 64 \
    --max_epochs 1000 \
    --precision bf16
```

**Expected timeline on 8x H200:**
- ~50-100 epochs for convergence
- ~12-24 hours total training time
- Your 10-40 hours of data is ideal for this architecture

### Phase 3: Add SLM Discriminator (Joint Training)

```bash
# Continue training with WavLM discriminator
python train_joint.py \
    --checkpoint matcha_epoch100.pt \
    --add_slm_discriminator \
    --slm_model microsoft/wavlm-large \
    --joint_epochs 200
```

### Phase 4: BigVGAN Fine-tuning (Optional)

```bash
# Fine-tune BigVGAN on synthesized mel-spectrograms
python train_vocoder.py \
    --pretrained nvidia/bigvgan_v2_24khz_100band_256x \
    --train_mels czech_mels/ \
    --epochs 100
```

---

## Alternative Approaches Considered

### F5-TTS
**Pros:** Zero-shot voice cloning, no phonemizer needed, SOTA quality
**Cons:** Requires 90+ hours for new language training (your 10-40h insufficient), implicit duration can cause minor alignment issues

### VITS (Coqui)
**Pros:** Pre-trained Czech model exists, simple training
**Cons:** Lower quality ceiling than flow-matching models, less expressive

### GPT-SoVITS
**Pros:** Excellent few-shot cloning, your data amount is ideal
**Cons:** Requires adding Czech phoneme system, some hallucination risk from AR component

### Training StyleTTS2 ASR from Scratch
**Pros:** Would give full StyleTTS2 quality
**Cons:** Requires separate Czech ASR corpus (100+ hours), complex multi-stage process, 2-3 weeks additional work

---

## Expected Quality

Based on comparable single-speaker training scenarios:

| Metric | Expected Value | Notes |
|--------|---------------|-------|
| MOS | 4.3-4.5 | Near human level |
| WER | Relative to the narrator's own recordings | The best open cs/sk ASR errs on 5.5–11% of human speech, so absolute 2–4% is unmeasurable (v2) |
| Speaker Similarity | 0.85-0.92 | High similarity to target |
| Hallucination Rate | Minimised structurally; gated per sentence | Explicit durations prevent AR-style drift; ASR verify-and-retry catches the rest (v2) |
| RTF | 0.05-0.15 | Real-time capable |

---

## Hardware Recommendations

For H200/B200 training:

```yaml
# Optimal configuration
gpus: 8
batch_size_per_gpu: 64-128  # H200 141GB VRAM
gradient_accumulation: 1-2
precision: bf16  # H200/B200 optimal
segment_length: 16384 samples

# Memory usage estimate
acoustic_model: ~20GB
discriminators: ~15GB
optimizer_states: ~30GB
data_loading: ~10GB
# Total: ~75GB per GPU (comfortable on H200)
```

---

## Summary

For maximum quality Czech TTS with voice cloning from 10+ hours:

1. **Use Matcha-TTS** as base - explicit durations, no ASR dependency
2. **Add an SSL discriminator** (StyleTTS2 idea, multilingual XLS-R-300M backbone in v2); keep it only if an ablation shows it helps
3. **Use BigVGAN v2** - SOTA vocoder quality
4. **Use espeak-ng** for Czech/Slovak G2P - native support, plus lexicon and post-rules (v2)

This gives you most of StyleTTS2's quality without its dependency problems. Explicit durations make hallucinations structurally rare, and ASR verification of every sentence (plan v2) catches the remainder.

The 10+ hours of data you have is ideal for this approach - enough for high speaker similarity, but not requiring the 100K+ hours that pure zero-shot models need.

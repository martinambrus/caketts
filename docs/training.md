# Pronunciation review and acoustic smoke training

These commands use the existing eSpeak NG phonemizer, Czech/Slovak post-rules and optional phonetic lexicon. G2P describes the text-to-phoneme step; no replacement backend is introduced.

The implemented runner is stage 2 acoustic Matcha-TTS training on one CPU or CUDA device in FP32. It supports bounded runs, validation and checkpoint resume. Mixed precision, distributed training, EMA, vocoder fine-tuning, adversarial stages and waveform evaluation are not implemented by this runner. A successful short run verifies training plumbing; it does not establish narration quality, useful MAS alignment or the final tempo policy.

## Prepare pronunciation review and vocabulary

Run from the repository root with the prepared manifests. On a CUDA environment use `uv run --no-sync` to retain the installed torch build.

```bash
mkdir -p review
uv run --no-sync scripts/g2p_review.py --manifest data/manifests/train.jsonl --out review/pronunciation.tsv
uv run --no-sync scripts/build_vocab.py --manifests data/manifests/train.jsonl data/manifests/val.jsonl data/manifests/test.jsonl data/manifests/heldout.jsonl --out data/vocab.json
```

The TSV contains private source words, counts, eSpeak phones, selection reasons and example context. It selects frequent words plus name, language-span and pronunciation-pattern candidates. It proposes no spelling, grammar or ASR correction. Check foreign names and unusual pronunciations first. Isolated words can devoice differently at a pause; the exporter includes actual token context where voicing matters. English phrases also have phrase-level review. Casing and pattern flags identify review candidates, not confirmed errors.

Manual pronunciation corrections belong in a phonetic JSON lexicon with language keys (`cs`, `sk`, `en`) and word-to-space-separated-phone mappings. This is separate from the corpus preparation orthographic replacement TSV. If a lexicon exists, pass the same `--lexicon lexicon/lexicon.json` to both tools and set `data.lexicon` in the training YAML. Changes to token counts require regenerating the manifest with the same phonemizer; the tools and loader fail on disagreement rather than cropping or silently remapping symbols.

Optional `--coverage cs=coverage/cs.txt` (repeat for other supported languages) adds normalized coverage sentences to the vocabulary. Lexicon-entry phones are included even when their words are absent from the manifests. No coverage text or phonetic correction is invented automatically. Special quote and continuation tokens are always present. Every manifest row is validated with strict symbol lookup and its quote context. Validation/test text can extend the symbol inventory; their audio is never used for optimization or training normalization.

The vocabulary stays a plain symbol-to-ID JSON file. Its `.meta.json` sidecar records input hashes, G2P provenance and the vocabulary digest. Existing files require explicit `--force`; avoid rebuilding a vocabulary after training starts. Forced vocabulary and metadata publication is atomic per file, not transactional as a pair; compare the recorded vocabulary digest after an interrupted rebuild. Review exports are private and should not be committed.

## Configure and run a small acoustic job

Edit `configs/training/smoke.yaml` to point to the manifests, vocabulary and an empty output directory. Paths inside the YAML resolve relative to that YAML, not the shell directory. `configs/training/finetune_narrator.yaml` is the larger acoustic-run template. These stage files use supported flat `MatchaConfig` options; they do not inherit the nested model template or silently accept unsupported settings.

```bash
uv run --no-sync scripts/train.py --config configs/training/smoke.yaml
uv run --no-sync scripts/train.py --config configs/training/smoke.yaml --resume outputs/acoustic-smoke/latest.pt --max-steps 5
```

`--max-steps` is an absolute optimizer-step target, including on resume. Use only trusted local checkpoints: loading restores Python RNG and optimizer state with explicit pickle semantics. The checkpoint must be inside the configured output directory. A nonempty run directory requires explicit valid resume. Checkpoint publication replaces `latest.pt` atomically so a failed write preserves the previous checkpoint.

The runner checks train/validation identities, actual audio and duration, strict vocabulary and stored token counts before optimization. It never truncates one side of an audio/text pair. Batches respect padded mel-frame cost (`longest frames × batch size`), and an individually oversized item fails rather than silently exceeding the budget. Duration bounds and item limits select a small subset without changing source manifests. Scalar mel normalization statistics come only from the selected training set and are reused on validation and resume.

Tempo conditioning is deliberately constant during bootstrap, overriding existing per-row word labels in in-memory copies. The manifest remains unchanged. Keep the word/MAS comparison pending until a trained alignment is meaningful; a few optimizer steps or a random model cannot choose that policy.

The output directory contains resolved configuration, runtime provenance, normalization, numeric `metrics.jsonl` and `latest.pt`. Logs include losses, gradient norm and MAS duration diagnostics; they contain no book text. Checkpoints bind configuration, vocabulary, lexicon, G2P identity, manifests, actual audio and selected samples, and restore optimizer, sampler cursor and RNG. Incompatible resumes fail before optimization. Keep run artifacts private.

Before a longer training run, review pronunciation choices and validate held-out audio with a vocoder and the planned quality gates. Acoustic loss checks alone do not certify word fidelity, pace or sound quality.

## Verification of the bounded implementation

The complete local feature suite passed 3,784 tests with four missing-speech-model skips (3,788 collected). A real-data CUDA smoke run used eight existing training clips and two validation clips with the small model, reached three optimizer steps, saved a checkpoint, and resumed to five total steps. Training/validation losses and gradients stayed finite. This is a pipeline and recovery check, not an audible voice test or an empirical tempo/MAS comparison. Source audio/text and corpus splits were preserved.

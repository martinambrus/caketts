# caketts
A TTS with style

An audiobook text-to-speech system for **Slovak, Czech and English**, in one narrator's voice. It pairs a Matcha-TTS core with a multilingual SSL discriminator and the BigVGAN v2 vocoder. It is trained on 10–40 hours of the narrator's recordings, with optional public cs/sk/en speech for pretraining.

It has two hard requirements:

1. **Every word is spoken once, in order.** No skipped or repeated words, no gibberish, no cut-offs.
2. **Steady pace.** The speed stays the same within a text and across texts.

## What's in the repository

| path | what it is |
|---|---|
| [`docs/czech_slovak_tts_implementation_plan.md`](docs/czech_slovak_tts_implementation_plan.md) | The implementation plan (v2, 25 September 2026): every step, with Claude Code prompts, acceptance criteria and tested code |
| [`docs/czech_tts_architecture_recommendation.md`](docs/czech_tts_architecture_recommendation.md) | Why this architecture |
| [`docs/research/tts_breakthroughs_since_late_2025.md`](docs/research/tts_breakthroughs_since_late_2025.md) | The research report behind the v2 changes |
| [`NUM2WORDS_CHANGES.md`](NUM2WORDS_CHANGES.md) | Slovak and Czech number-to-words v2: every change against v1, the sources, and the decisions from the native-speaker review |
| `src/`, `tests/`, `scripts/` | The reference implementation. These are the plan's tested code blocks as files. |
| `docs/research/num2words_spec_*.md`, `docs/NUM2WORDS_v1_v2_diff.csv` | Research notes and the full v1 → v2 diff for the number modules |

The first commit holds the v1 plan and the v1 number modules, so `git log -p` shows what changed in v2.

## Status

These parts are implemented and tested on CPU:

- Environment setup: the model config and the environment test
- G2P: espeak-ng with post-rules, a lexicon and a strict vocabulary
- Slovak and Czech number-to-words, with case, gender and animacy
- Audio preprocessing with BigVGAN's exact mel spectrogram
- The data pipeline
- The Matcha-TTS components: text encoder, duration predictor, MAS, OT-CFM decoder
- The SSL discriminator
- Tempo labels
- Pace metrics and the ASR check

These are still prompts with acceptance criteria in the plan:

- The text normalizer and sentence segmentation
- The G2P review and vocabulary scripts
- Corpus preparation and the tempo-labelling script
- The vocoder wrapper
- The training loops and stages
- Book-length inference
- The production API
- Evaluation and the QA report

## Quick start

```bash
uv sync   # Python 3.13 and the locked packages (CPU build of torch) into .venv; uv: https://docs.astral.sh/uv/
git clone https://github.com/NVIDIA/BigVGAN third_party/BigVGAN   # or: export BIGVGAN_DIR=/path/to/BigVGAN
git -C "${BIGVGAN_DIR:-third_party/BigVGAN}" checkout 7d2b454     # the commit the tests ran with
uv run pytest tests/ -q -m "not slow"   # 199 tests, about 15 s on CPU
uv run pytest tests/ -q                 # adds the end-to-end synthetic training test, about 40 s
```

The first test run downloads the text normalizer's Stanza models for Czech and Slovak (250 MB) into `~/.cache/stanza`. Stanza fetches them through huggingface_hub, which keeps a second copy in `~/.cache/huggingface/hub`; delete its `models--stanfordnlp--stanza-*` folders to free that space.

Numbers to words:

```python
from src.text.num2words_sk import num2words as sk
from src.text.num2words_cs import num2words as cs

sk(25, case="genitive")                      # 'dvadsiatich piatich'
sk(2, gender="feminine", case="instrumental")  # 'dvomi'
cs(1991, to="ordinal")                       # 'tisíc devět set devadesátý první'
cs(3.14)                                     # 'tři celé čtrnáct setin'
```

To print every form for review, run `uv run scripts/validate_all_sk.py` or `uv run scripts/validate_all_cs.py`. `uv run --with PyICU scripts/cldr_crosscheck.py` compares both modules with the Unicode CLDR rules; PyICU builds against ICU, so install `libicu-dev` first.

## License

MIT; see [LICENSE](LICENSE).

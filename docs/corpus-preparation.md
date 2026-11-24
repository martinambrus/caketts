# Corpus preparation: from a narrated book to training clips (Step 4.2)

This turns the recordings of a book and its text into a training set: clips of one sentence (or one quote) each, with the normalised text, the pause that follows, the audit verdict of every clip, and a manifest split into train, val, test and one held-out block. Reports say what was dropped and why. One entry point, `scripts/prepare_corpus.py`, runs the stages from one YAML file. Each stage reads only the files of the earlier stages in the work directory and writes its own, so the same inputs give the same files (the timings in the reports aside) and any stage can be re-run alone.

It was built and measured on one Czech audiobook of 9.09 hours (10 parts, one narrator): 6.61 hours were kept. The numbers in this guide come from that run, and only Czech has been run on a real book. The plan's Step 4.2 holds the design decisions; this guide is for using it.

## What you need

- **Recordings:** one WAV file per part, `raw/<part>.wav`. A part is a chapter or a session; the stages stream the audio, and parts of 26 to 60 minutes were used. Any sample rate; stereo is mixed to mono. Name the parts so that their natural order (`a2` before `a10`) is the book order: the split and the held-out block follow it. Keep a part under about 12,000 words (roughly 2 hours of narration): the align stage matches the text words to the ASR words in a table of at most 150 million cells and stops with `ValueError: split the part` above that.
- **The text of each part, as the narrator read it.** The pipeline never repairs the text from the audio. Where the narrator departs from the book, the clip is dropped (about 41% of the clips the audit dropped in the first book). Text the narrator read is worth more than any later fix: see "Choosing the text" below.
- **A GPU is recommended.** On one 60-minute part the ASR pass took 2 minutes on a 6 GB NVIDIA GPU; on the CPU of the same machine the ASR pass took 17 minutes and the alignment 38 more. The whole first book took 42 minutes on the GPU.
- Python 3.13 and [uv](https://docs.astral.sh/uv/); `espeak-ng` comes with the project through `espeakng-loader`.

## Install

```bash
uv sync --group align --group cu130 --no-group cpu   # NVIDIA GPU: CUDA 13 build of torch (about 5 GB more than the CPU build)
uv sync --group align                                # CPU only
```

The `align` group adds `uroman`, which only the `mms-300m` aligner needs. A plain `uv run` syncs the default groups again and would put the CPU build of torch back, so in a CUDA environment run `.venv/bin/python scripts/prepare_corpus.py ...` (or `uv run --no-sync ...`). The models are downloaded from the Hugging Face hub on first use into its cache; set `HF_HOME` to choose where.

## Configure

```bash
cp configs/data/example.yaml book.yaml
```

`example.yaml` lists every key at its default with a one-line meaning; delete what you do not use. Paths are relative to the YAML file. A Czech book with `raw/a1.wav`, `raw/a1.txt.orig`, `raw/a2.wav`, ... needs only:

```yaml
language: cs        # cs or sk; fed to every stage that needs it
raw_dir: raw
work_dir: work
align:
  backend: xls-r-cs   # see "Choosing the aligner"
```

`language`, `speaker` and `use_part_fragments` are set once at the top level. An unknown key raises an error that names its path. The text files default to `raw/<part>.txt.orig`; `text.suffix` changes the suffix.

## Run

```bash
python scripts/prepare_corpus.py <stage> --config book.yaml     # one stage
python scripts/prepare_corpus.py all --config book.yaml         # the stages the config asks for, in this order; stops at the first failure
```

| stage | reads | writes (in `work_dir`) |
|---|---|---|
| `source` (optional) | the EPUB, your texts | `source/epub/<part>.txt`, `source/epub/placement.json`, `source/orig/placement.json`, `source/report.*` |
| `lexicon` (optional) | `source/epub/*.txt`, your rewrite | `lexicon.tsv` |
| `text` | the book texts | `text/<part>.norm.txt`, `text/<part>.units.jsonl`, `text/review_readings.csv`, `text/report.*` |
| `audio` | `raw/<part>.wav` | `audio/<part>.24k.wav`, `audio/<part>.16k.wav`, `audio/<part>.json`, `audio/report.*` |
| `align` | units, 16 kHz audio | `align/<part>.align.jsonl`, `align/<part>.meta.json`, `align/<part>.asr.json`, `align/report.*` |
| `cut` | units, alignment, 24 kHz audio | `cut/clips/<id>.wav`, `cut/<part>.cuts.jsonl`, `cut/<part>.rejects.jsonl`, `cut/pauses.json`, `cut/report.*` |
| `audit` | the clips | `audit/<part>.audit.jsonl`, `audit/<part>.asr*.jsonl`, `audit/review.csv`, `audit/listening.csv`, `audit/report.*` |
| `manifest` | all of the above | `manifest/{train,val,test,heldout,dropped}.jsonl`, `manifest/report.*`, `manifest/coverage.*` |
| `gold` (optional) | clips, alignment, a gold file | `gold/report.json`, `gold/report.md` |

Every stage takes all parts of the config in one call. A missing input prints one error line that names the stage to run first. A report holds counts, ids and numbers, never the book's text. Most other files of the work directory do hold text (the book texts, `lexicon.tsv`, the alignment, the clip texts, the audit's transcripts, the manifests, the review lists), so keep the work directory private. Look at each stage's report before running the next; the checks below say what to look for.

### Choosing the text

| | the narrator's version of the text | EPUB + a lexicon learned from it |
|---|---|---|
| units aligned `ok` | 94.7% | 75.7% |
| clips dropped by the audit | 11.9% | 22.3% |
| audio kept in the manifest | 70.9% | 56.1% |

Measured on two parts of the first book (1.43 hours). 81% of the EPUB failures were text the narrator skipped or changed (chapter numbers, one-word replies, speaker labels, printout lines); text-side fixes (a speaker-label rule, tables, names, numbers) would reach about 79% `ok` at most. So there are two routes:

1. **A text that matches the narration** (the book as the narrator's text, or your own rewrite of it): put `raw/<part>.txt.orig`, one file per recording. This is what the first book used. Chapters are then unknown; add them from an EPUB (the `source` section below) or list heading lines in `text.headings`.
2. **The EPUB and a lexicon.** The EPUB gives the book's text with its chapters; the lexicon maps the forms it prints to the forms the narrator spoke.

### source (optional): the EPUB as the book text

```yaml
source:
  epub: {path: book.epub}
  orig_dir: raw                  # your texts mark where each part starts and ends in the EPUB
text:                            # route 1: your texts, with the EPUB's chapters
  text_dir: raw
  suffix: .txt.orig
  placement: work/source/orig/placement.json
```

The EPUB's paragraphs are cut into the parts you recorded, at the word (a part border often falls inside an EPUB paragraph: 5 of 9 did in the first book). Instead of `orig_dir`, `source.parts` lists `{name, start}` with `start` an EPUB item index or the first words of the part. A paragraph that matches `epub.chapter_regex` (default: `12 / Title`) becomes the heading `# 12. Title`. Paragraphs that hold a TAB (tables) are left out and listed unless `keep_tables: true`. `replacements` applies literal replacements to the book text. With `orig_dir` the stage also writes `source/orig/placement.json`, which places the EPUB's chapters into your own texts (route 1). With a `source` section and no `text` settings, the text stage reads `source/epub/<part>.txt`, the EPUB's own text (route 2).

### lexicon (optional): the forms the book prints but the narrator did not say

Needs the book text (`source/epub/<part>.txt`) and a rewrite of it that matches the narration (`lexicon.rewrite_dir`, files `<part><rewrite_suffix>`). The stage aligns both word by word and writes `lexicon.tsv`: every difference of at most 6 words per side (30 when the book side holds a symbol the normalizer cannot read) as a pair `book_form`, `spoken_form`, with a category (symbol, respelling, number, abbreviation, insertion, deletion, other), a count, `consistent` and an `apply` column (`y` or `n`). The text stage replaces every applied book form, as a whole word or phrase, before it normalises. `apply` starts as `y` for a consistent respelling (a name or foreign word the rewrite always writes the same way) and, for a symbol, for its most frequent spoken form, and as `n` for everything else (numbers, abbreviations, insertions, deletions): those are left to the normalizer, and `text/review_disagreements.csv` lists where its reading differs from the rewrite.

The stage overwrites `lexicon.tsv` on every run. To keep hand edits, copy the file and point `text.lexicon` at the copy.

A hand-written lexicon is the same file with a header `book_form`, `spoken_form`, `apply` (TAB separated, one applied row per book form, `apply` `y` or `n`); an empty file is an empty lexicon. Use it for names, foreign words and symbols the narrator read in a fixed way.

Route 2 in full (the EPUB, your texts as the part borders and as the rewrite, a lexicon, unreadable paragraphs skipped):

```yaml
source:
  epub: {path: book.epub}
  orig_dir: raw
lexicon: {rewrite_dir: raw}
text: {skip_unreadable: true}
```

### text

One file per part: paragraphs separated by blank lines, a line that starts with `# ` is a chapter heading, and a text with no heading is one chapter. The stage reads the bytes, converts the line endings and reports the format (BOM, line endings, tabs, no-break spaces, blank runs, consecutive non-blank line pairs, heading candidates such as short unfinished or all-capital lines), but never repairs it. Chapters come from `# ` lines, from `text.headings: {a1: [1, 40]}` (line numbers that are headings) or from a `placement.json`.

The whole part goes through the normalizer in one call, since its readings depend on the neighbouring paragraphs. Then it is segmented (Step 2.3) and cut into **units**: one sentence each, except that a quote or bracket still open at the end of a sentence keeps the next sentence of the same paragraph while the estimated length of the unit stays within `max_est_s` (17 s); a longer unit is cut after a clause mark (, ; : — – … or a spaced hyphen). A unit never crosses a paragraph. The same function, `src/text/units.py`, renders the units at inference (Step 12). Each unit carries tags (end mark, quote and narration order, paragraph and chapter position, number, date, abbreviation, ...) that drive the coverage report and the split.

- A character the normalizer cannot read stops the stage and prints the failing lines. Fix the text, add a lexicon row, or set `skip_unreadable: true` to blank those paragraphs and list them in `text/report.json` (`skipped_blocks`); the narrator's audio of a blanked paragraph then falls into a gap at the align stage.
- `review_readings.csv` lists every reading the normalizer chose by a guess, with part, paragraph and line. Check the ones a listener would notice (numbers, abbreviations, names).
- Check in `report.md`: `units_over_cap` (0 on the first book), the heading candidates and the tag counts of each part.

### audio

Per part, streamed in 30 s blocks: a mono mix; loudness -23 LUFS over the whole part (EBU R128), with a peak limiter taking the few peaks above 0.99 instead of lowering the gain of the whole part (6 of the 10 parts of the first book needed it); 24 kHz (the clips) and 16 kHz (aligner and ASR) 16-bit WAV files. `audio/<part>.json` records the input rate, channels, loudness in and out, gain and peak.

### align

CTC forced alignment of each unit's words to the audio with a wav2vec2-style model. A part longer than `block_max_s` (600 s) first gets an ASR pass with word times (`align.asr`, Parakeet-TDT-0.6B-v3); its words are matched to the text by an edit-distance alignment, the matched words give every text word an estimated time, and the units are grouped into blocks of at most 10 minutes (each with 3 s of audio on both sides) that are aligned one by one. The ASR words are cached in `align/<part>.asr.json` (keyed by the audio, the model and the window settings), so a re-run does not repeat the pass. Text the narrator skipped and speech that is not in the text become gaps (`skipped_text`, `extra_speech`, in `align/<part>.meta.json`).

Each unit gets a status from the part's own score distribution: `ok`, `low_conf` or `unmatched`. Only `ok` units are cut; the others are listed and never become clips. The aligner's spans lie inside the speech (against the gold, the start about 60 ms late and the end about 90 ms early with `xls-r-cs`), so the cut stage refines them by energy.

Check: the `ok` count against the total in `align/report.md` (4,721 of 4,952, 95.3%, on the first book with `xls-r-cs`). A low share means the text differs from the narration or the aligner does not fit the language: read the text at the gaps (`align/<part>.meta.json`), and try another `align.backend`.

With `align.edge_repair_max_s: 0.6`, an optional correction recovers short first words that CTC placed entirely after their ASR interval. It requires a unique run of three matching words, agreement of the next two word intervals, audible source audio and room after the previous unit. It changes only already-`ok` units and never promotes rejected units. The default is `0.0` (disabled); validate the enabled setting on the book with gold scoring and the clip audit. It does not repair arbitrary alignment errors.

### cut

Per part, in integer samples, with every number in `cut:` of `example.yaml`:

1. The speech edge of each unit is moved from the aligner's span to the energy edge (10 ms frames, speech = at least 15 dB above the part's noise floor, the 5th percentile of the frame levels), searched within 80 ms. A dip of at most 30 ms is bridged; a dip of up to 0.35 s (a breath, or a first word the aligner started late) is bridged to a burst of speech-level audio beyond it, only while a quiet stretch remains between the two units.
2. The cut point between two clips is the centre of the longest quiet run (at least 40 ms, at most 15 dB above the floor) inside the measured gap. Each clip keeps about 150 ms of silence on each side (the pad), never past the cut point. The pad edge is placed 150 ms from the speech edge; when the 10 ms frame there is louder than the quiet level (a breath or noise sits at it), the edge moves outward, one frame at a time, to the first quiet frame, at the furthest to the cut point, and never so far that the clip passes the 20 s cap (where it would, the edge stops at the cap and stays flagged). A pad edge is never moved toward the speech, so the speech edges, the flags and `cut/pauses.json` do not depend on this step. An edge never moves past the midpoint of a word of the align stage's ASR pass (`align/<part>.asr.json`), so an extended pad never takes in a word of the neighbouring unit that the aligner placed late (a late first word): such an edge keeps its place and stays flagged. A part with no ASR pass (shorter than `align.block_max_s`) has no such guard.
3. A gap shorter than two pads (0.3 s): the two units merge into one clip when they are in the same paragraph (the first ends mid-paragraph, neither is a heading or a piece of one split sentence), the aligner's own gap is under 0.46 s, and the joined clip is estimated within 17 s and measures at most 20 s with its pads; otherwise the cut is made at the quiet centre, each pad shrinks to fit, and the clip gets the flag `short_pause`. No quiet run in the gap: the cut is made at its middle and both clips get `no_quiet_cut`. A clip that measures over 20 s is split at its longest word pause (at least 3 words on each side); where that is impossible it is flagged `over_cap`.
4. Hard assertions per part: every clip starts before it ends, no two clips overlap, a clip holds its words, the bounds lie inside the part, and every clip's PCM hash is unique over the corpus. A violation stops the stage.

`cut/pauses.json` holds the narrator's median pauses by type (comma, sentence, paragraph, chapter), measured from speech edge to speech edge, and `pad_s`, the nominal pad (a clip carries about that much: less in a short gap, more where noise sat at its edge). Step 12 inserts pauses from it. A breath next to the speech counts as speech, so the medians are breath-free (sentence 0.95 s on the first book).

### audit

Two independent judges transcribe every clip (`audit.judges`: Parakeet-TDT-0.6B-v3 and Canary-1B-v2; neither may be the aligner: the config load checks that the names are valid and distinct and that the aligner differs from the ASR anchor model, and the audit stage checks every judge against the aligner model recorded in `align/<part>.meta.json`). A judge's CER (digits spelled out) is compared with its own corpus median plus `cer_margin` (0.05), and a clip of at most 30 characters needs at least 3 character edits. **A clip is dropped only when every judge exceeds its threshold** (`cer`); when only some do, it is kept and listed (`cer_disputed`). With one judge the rule is the single-judge rule; on two parts of the first book it dropped 22.2% against 12.9% for two judges, and about half of the single-judge drops were false (numbers, short interjections, recognition errors).

Other findings go to `audit/review.csv` (clip, metric, value, threshold, reason) without dropping: speaking rate outside 3 robust sigmas, noise at a clip edge (an edge frame louder than the cut's quiet level `cut.quiet_db`, unless `audit.edge_db` is set; the cut already moves a noisy pad edge to a quiet frame, so this is left for the few clips where that was not possible), clipping, loudness, and the cut flags `no_quiet_cut` and `over_cap`. `audit/listening.csv` ranks the 50 riskiest clips for a listening pass; listen to it before training. The transcripts are cached by clip hash, model and requested book-language/control identity, so after a re-cut only changed or incompatible clips are transcribed again. Known legacy caches require matching producer, part and audio evidence; a missing language is never assumed to mean the current book language.

### manifest

Clips the audit dropped, over-long (> 20 s), too short (< 0.5 s) and impossible clips (fewer mel frames than tokens, Step 4.3) are left out and listed with their reason in `dropped.jsonl`. The rest is split:

- **val and test** take whole paragraphs to fill a quota per tag (the smaller of 5 clips and 10% of the tag's clips, rounded up, rarest tags first; a tag with fewer than 3 clips has no quota). A paragraph with a clip flagged `short_pause`, `no_quiet_cut` or `over_cap` stays out of val and test (its clips can still train), so the evaluation sets have uniform clip edges; `manifest.exclude_flags` and `exclude_tags` change that.
- **held-out** is one block of consecutive paragraphs of one chapter, about 4% of the hours, for long-text and pace checks: a whole chapter of 0.5 to 2 times that share or, when none fits, a window of paragraphs that reaches the share and stays within twice it (with none, there is no held-out block). Clips the audit dropped and units the align stage rejected leave holes in the text of a block, and a hole-free block of that size cannot exist in a real book, so among the candidates the one with the smallest share of holes between its clips wins, then the one whose tags are closest to the corpus's; `manifest/report.*` lists the holes of the chosen block.
- **train** is the rest, minus any clip whose text (4 words or more) equals a val, test or held-out text.
- The checks run on every manifest: no paragraph in two splits, no identical audio in two splits (hashes recomputed from the WAV files), held-out consecutive.

Each manifest row has `id`, `audio` (the absolute path of the clip), `text`, `lang`, `speaker`, `session` (the part), `boundary`, `duration_s` (from the written WAV), `articulation_rate`, `pause_ratio` (from the aligner's word times, first to last word, pads excluded), `gap_before_s`, `gap_after_s`, `tags`, `split`, the audit verdict and CER, `units`, `paragraph`, `chapter`, `flags`, `n_tokens` and `sha256`. The rows load through `load_manifest`, `filter_manifest`, `TTSDataset` and `TTSCollator` (Step 4.3). `coverage.md` counts clips and hours per tag and split and lists thin tags; read it to see which kinds of sentence the model will never see.

Quote continuation is explicit: `quote_before` and `quote_after` carry quote-only context, independently of the combined quote/bracket depths. The shared phonemizer accepts these fields and emits continuation context tokens without adding quote marks to the text. Re-run text, cut and manifest and rebuild the vocabulary to add this context to an older corpus. The dataset checks stored token counts against its actual phonemizer and uses mel settings as part of the cache key. A phonetic lexicon must be supplied through the same G2P used by manifest generation and training; `text.lexicon` is a separate orthographic-replacement TSV.

Cut and gold scoring verify a long-form ASR cache against its 16 kHz source audio. An existing cache with a missing or mismatched audio hash stops the stage; re-run alignment to regenerate it. A missing cache remains optional.

### Book language and terminology

Set `language: cs` for a Czech book or `sk` for Slovak. Shared and ambiguous words inherit that book language; an existing explicit `<en>...</en>` span declares an exception. Canary receives the book language as both source and target. Parakeet remains automatic, which its recorded provenance states; a book setting does not create an unsupported forcing mechanism. Neither backend's recognition errors are authority to standardize the narrator's grammar.

Set `text.terms: terms.tsv` for persistent book choices. The audit inherits this file. Its tab-separated columns are `canonical`, `alias`, `language`, and optional `kind`:

- `name` (default) protects a narrated name form. List ASR spelling aliases on additional rows with the same canonical form.
- `protected` preserves a source form, including spoken colloquialisms that ASR changes into standard grammar.
- `word` admits a reviewed ordinary vocabulary item.

An alias may be blank. Entries are exact word forms; enumerate genuine inflections separately. Language declares the source context and does not change G2P: pronunciation still uses the existing inline spans, `book_config.english` and phonetic lexicon. Deliberate book-wide source respellings use the existing orthographic `text.lexicon`.

The corrected source vocabulary, capitalized forms and explicit spans provide conservative protection across every text part, even for a command selecting fewer parts. This is neither a dictionary nor a foreign-word detector. Unfamiliar or competing forms are grouped for a book-level decision instead of being guessed into another language. The aim is fidelity to the recording, including slang and colloquial grammar.

Declared aliases produce a separate diagnostic comparison only when a one-word substitution aligns with that exact canonical source form. Raw judge transcripts and the original CER quality gates remain unchanged; insertions, deletions and repeats are not excused. The policy file and every consulted source part are bound into proposal provenance. After changing a choice, re-run the audit and regenerate proposals; compatible raw transcripts can be reused. Write regenerated proposals to a new filename so prior review notes are retained.

### Reviewed transcript repairs

When narration differs from the transcript, generate suggestions from the saved judge transcripts:

```bash
uv run --no-sync scripts/repair_corpus.py propose --config corpus.yaml --proposal repairs.json
```

The proposal contains ordinary word substitutions in audit-dropped clips, supported by every configured judge and the trusted book vocabulary; casing-only differences are ignored. Names, protected spoken forms, new vocabulary and competing choices are grouped in `term_review`. Weak single-judge guesses remain in the audit evidence and are not promoted into either decision queue. Listen before setting an edit's `reviewed` to `true` and filling its `reviewer`. Unreviewed suggestions are not applied. The tool handles equal-word-count substitutions in unsplit single-unit clips; insertions, deletions and merged clips still need transcript editing. It preserves punctuation, dialogue marks and language spans.

```bash
uv run --no-sync scripts/repair_corpus.py apply --config corpus.yaml --proposal repairs.json --output repaired-work
```

The source configuration, book policy and input hashes must still match. Apply recomputes eligibility from the current evidence; a grouped item cannot be promoted by editing its JSON review fields. Update the book policy or orthographic lexicon and regenerate instead. Apply creates a separate candidate with an original-text snapshot and receipt, then runs alignment, cut, audit, manifest and configured gold scoring. A completed receipt means those stages ran; inspect their reports before accepting the candidate. Source recordings and the previous corpus remain available.

### gold (optional): checking the cut against speech you verified

The scorer takes a gold file (`gold.file`, JSON: verified speech spans and pause ranges per part, see `src/data/prepare/gold.py`) or one Audacity label file per part (`gold.audacity: {a1: labels-a1.txt}`): in Audacity, label each pause between two sentences, from the end of the last word to the start of the next, and export the labels (TAB separated: start, end, label; seconds). Labels give the pause-centre error only; verified speech spans also give the speech-edge error and the main criterion, **no speech cut off**: a matched span that loses an ASR-confirmed word (an ASR word within 50 ms of speech-level audio that no clip holds). Words of rejected units are never cut, by design, so they do not count: a lost word that lies between two `ok` units with a rejected unit between them, less 80 ms (the cut's search window) at each `ok` edge, is reported apart as a rejected-unit word (`rejected words` in `gold/report.md`). On the first book (248 verified spans) it is 1 of 153 matched spans (0.7%): a first word the aligner placed late, which no clip holds; 16 words of rejected units in 4 spans are listed apart (counted as lost, it was 2 of 153). The unsafe-cut rate (a clip edge strictly inside a gold span) is reported for information only, since a gold span can hold pauses of its own.

## Comparing tempo labels before choosing a policy

The word-alignment labels in the manifest exclude clip pads. A MAS export must come from a meaningful trained checkpoint and use the same tokenization, quote context and audio hashes. Export JSONL rows with `id`, `sha256`, `kinds` and `durations_frames`, excluding collator padding. The helper checks exact clip IDs, token counts and mel-frame totals, removes clip padding even when it spans multiple tokens, and compares both quote-duration policies without changing the labels:

```bash
uv run --no-sync scripts/compare_tempo.py --manifest manifest/train.jsonl --mas trained-mas.jsonl \
  --cuts cut/a1.cuts.jsonl cut/a2.cuts.jsonl --hop-s 0.010666666666666666 --output tempo-comparison.json
```

Supply all cut files used by that manifest. The report contains numeric deltas and per-session and per-chapter medians; it does not contain transcript text. One policy counts quote-token durations as speech, the other counts durations of at least 120 ms as pauses. Inspect the comparison before selecting a policy or replacing labels. No trained-MAS comparison has been run yet.

## Choosing the aligner and the judges

Every model behind a stage is chosen by config; the weights keep their own licences. A CC-BY-NC weight is yours to accept per run: the table records the licence, it does not decide.

| role | `backend` | hub id | weights licence | model file |
|---|---|---|---|---|
| aligner | `mms-300m` (code default; needs the `align` group) | MahmoudAshraf/mms-300m-1130-forced-aligner | CC-BY-NC-4.0 | 1,262 MB |
| aligner | `xls-r-cs` (Czech only) | comodoro/wav2vec2-xls-r-300m-cs-250 | Apache-2.0 | 1,262 MB |
| aligner | `mms-1b-all` | facebook/mms-1b-all | CC-BY-NC-4.0 | 3,860 MB and a 9 MB language adapter |
| ASR (anchors) and judge | `parakeet-tdt-0.6b-v3` | nvidia/parakeet-tdt-0.6b-v3 | CC-BY-4.0 | 2,508 MB |
| judge (no word times, so no anchors) | `canary-1b-v2` | nvidia/canary-1b-v2 | CC-BY-4.0 | 3,916 MB |

All five take about 12.8 GB of disk; the recommended Czech setup (`xls-r-cs`, Parakeet and Canary) takes 7.7 GB. NVIDIA Forced Aligner is not used: it takes CTC or hybrid models only, and Parakeet-TDT is neither. The ASR models run through `transformers`, not NeMo. `uroman` (group `align`) is under a permission text that asks to acknowledge the software in publications; the PyPI classifier says Apache, and the text in the package is what applies.

The same text, audio, ASR cache and cut settings with only `align.backend` changed (first book, 5,072 units before the unit re-fit, the gold column still counting the words of rejected units as lost; the GPU figures are the memory in use, including about 1.5 GB of other programs):

| aligner | units `ok` | low_conf / unmatched | matched gold spans | matched spans losing a word | clips | align time, warm ASR cache | peak RAM / GPU |
|---|---|---|---|---|---|---|---|
| `xls-r-cs` | 95.4% | 223 / 10 | 158 | 1.27% | 4,698 | 401 s | 2.78 GiB / 3,039 MiB |
| `mms-1b-all` | 91.3% | 356 / 87 | 152 | 1.32% | 4,509 | 706 s | 5.37 GiB / 4,437 MiB |
| `mms-300m` | 86.5% | 490 / 196 | 140 | 2.14% | 4,265 | 476 s | 2.78 GiB / 3,066 MiB |

The refined clip edges are within 10 ms of the gold (median) for all three, since the cut refines them by energy. `xls-r-cs` is recommended for Czech; for Slovak the choice is `mms-300m` or `mms-1b-all`, which the config's `language` adapts. Compare them on your own book: the `ok` share and the number of rejected units tell which fits.

## Disk, memory and time

- Models: see the table above. Python environment with the CUDA build: about 6 GiB.
- The run directory of the first book: 3.7 GiB (`audio/` 2.5 GiB, both sample rates of every part; `cut/clips` 1.2 GiB; everything else under 10 MB). The raw recordings: 5.4 GiB.
- Whole book, cold, one process on a 6 GB GPU (an RTX 3050): 42 min 25 s, peak RAM 5.46 GiB. GPU memory in use peaked at 5,199 MiB during the audit, about 1.5 GB of it by other programs.

| stage | wall time | peak RAM, GiB |
|---|---|---|
| source | 2 s | 0.59 |
| text | 7 s | 1.34 |
| audio | 3 min 08 s | 1.43 |
| align (ASR of 10 parts, then CTC) | 21 min 05 s | 4.07 |
| cut | 1 min 14 s | 3.15 |
| audit (Parakeet and Canary, 4,586 clips) | 16 min 27 s | 4.55 |
| manifest | 4 s | 3.46 |
| gold | 8 s | 3.60 |

After changing the cut settings, re-run `cut`, `audit`, `manifest` and `gold` only: on the first book that took 1.5 min, 1 min (changed clips only), 8 s and 9 s.

## Results on the first book

Baseline before the optional first-word correction: Czech, 10 parts, 9.091 hours, text = the narrator's rewrite with the EPUB's chapters, `xls-r-cs`, two judges:

| step | units or clips | hours |
|---|---|---|
| recordings | 10 parts | 9.091 |
| text units | 4,952 (56,791 words; 24 clause cuts; none over the cap) | |
| aligned `ok` | 4,721 (95.3%); 221 `low_conf` and 10 `unmatched` rejected | |
| cut clips | 4,586 (129 merged; 8 re-split pieces); longest 19.39 s | 7.364 with pads |
| dropped by the audit | 501 (10.9%); the causes of the 499 drops before the clip-edge change: narrator deviation 207, number forms 134, short clip 120, judges disagree 38 | 0.751 |
| manifest | 4,084 | 6.613 (72.7% of the recordings) |
| train / val / test / held-out | 3,819 / 42 / 49 / 174 | 6.107 / 0.094 / 0.113 / 0.300 |

The held-out block is one chapter of 58 paragraphs (10 of its units missing: 8 dropped clips and 2 rejected units). One test quota (parenthesis) could not be met. Cut flags: `no_quiet_cut` 83 clips (1.8%), `short_pause` 322 (7.0%), `over_cap` 0. Pad edges on a frame louder than the quiet level: 7 of 9,172 (211 pads extended outward, 4 kept by the ASR-word guard). Pause medians: sentence 0.95 s, paragraph 1.73 s, chapter 2.93 s, comma 0.10 s. The text has no digits after normalisation, so there are no number or date clips. The checks passed: 4,586 distinct clip hashes and no overlaps, 0 shared audio between splits, manifest invariants, the round trip through `TTSDataset` and `TTSCollator` for all 4,084 rows (2,186 quote tokens), and no book text in any report.

The cut defaults were tuned on the gold of 8 parts and checked on 2 held-out parts (quiet level 12 to 15 dB and the guarded bridge to speech-level audio): the gold spans losing a word fell from 17 to 5 of 246, the median start-edge error from 60 to 10 ms, and the clips with noise at the start edge by about 60%, at the cost of more `short_pause` clips (178 to 382: a breath absorbed into the speech shrinks the measured gap) and pause medians that measure breath-free silence (sentence 1.13 s to 0.93 s).

The unit-length estimate (`estimate` in the text stage, which decides the 17 s cap and the merges) was re-fitted on the measured clips of the whole book: error on the spoken seconds of 4,574 single-unit clips went from bias -1.33 s and RMSE 1.68 s to bias 0 and RMSE 0.89 s; fitted on parts a1 to a5, the RMSE on a6 to a10 is 0.90 s.

### Candidate with anchored first-word correction

With `align.edge_repair_max_s: 0.6`, replaying the correction over the saved CTC outputs and hash-verified source audio changes 354 first-word spans. A real-model GPU excerpt reproduced the original failing boundary. This is a reuse of existing CTC outputs, not a new whole-book CTC run. Text-stage output is unchanged apart from the new context fields.

The candidate has 4,582 cut clips (four additional merges), of which 298 have changed audio hashes and 4,284 are unchanged. The longest remains 19.39 s. The normal two-judge audit transcribed the changed clips again (596 new transcripts): drops fell from 501 to 497; six changed clips went from drop to review and two from review to drop. Gold lost words fell from one to zero; the matched-span count changed from 153 to 152 after merging, with 245 covered gold spans unchanged. This does not establish that every possible word error is eliminated.

The candidate keeps 4,084 clips, 6.634 h: train/val/test/held-out remain 3,819/42/49/174 clips, with 6.127/0.095/0.111/0.300 h. All rows load through the dataset and collator with matching counts, a 64-symbol vocabulary, and 225 quote continuation tokens. Split checks pass. Six pad edges remain flagged, down from seven in the baseline. The optional correction stays disabled by default outside this validated candidate.

The reviewed-repair proposer finds 318 unanimous word substitutions in 185 dropped clips. No replacements have been applied; those suggestions still require listening verification. The original corpus and rollback backup are preserved.

### Narration-fidelity terminology pass

A subsequent metadata-only pass fixed book language/cache provenance and added the reviewed term policy. All 9,164 raw judge transcripts were reused with validated producer/part/audio evidence; no speech was re-decoded. The 4,084 manifest rows retain the same audio, text, labels and split assignments. The explicit policy supplied 18 contextual alias diagnostics without changing the raw CER gates.

The previous 318 unanimous suggestions became 42 ordinary edit suggestions and 276 occurrences grouped into 177 terminology items. Reported grammar/colloquial source forms were protected. The four suspected alignment examples showed no demonstrated cutting defect, so no audio boundaries were shifted for them. These queues are still unapproved; no transcript replacement has been applied.

## What it does not do

- It does not automatically accept ASR text. Reviewed word substitutions are supported; insertions, deletions and merged-clip repairs still need transcript editing. A completed candidate must pass its normal audit before training.
- It does not read what the normalizer cannot read: `|`, `•` and a `/` it cannot place (such as `139/VII`) stop the text stage; list them in a lexicon. The normalizer reads `/` between words ("or"), between numbers and in a ratio such as `Kč/kg`.
- Only Czech has run on a real book, and the aligner comparison exists for Czech only. The config accepts `language` `cs` and `sk` only (the normalizer's languages): English appears only as `<en>` spans inside a Czech or Slovak text.
- The manifest's rates and the pauses in `pauses.json` describe this narrator in these recordings; recompute them for another book.

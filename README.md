# Dementia Detection from Speech & Text — Starter Pipeline

A working, tested pipeline that extracts **linguistic** and **acoustic**
features from transcripts/audio and trains classifiers (Logistic Regression,
SVM, Random Forest) to distinguish AD vs. healthy control (HC) speakers.
It also includes a **TF-IDF text baseline** and a small **feature-selection**
step for the handcrafted feature models.

## Files
- `dementia_detection_pipeline.py` — main pipeline (feature extraction + training)
- `cha_to_txt.py` — converts DementiaBank/ADReSS `.cha` transcripts to plain `.txt`
- `requirements.txt` — dependencies

## Quick start (verify the code works, no dataset needed)
```bash
pip install -r requirements.txt
python dementia_detection_pipeline.py --demo
```
This runs on synthetic Cookie-Theft-style transcripts so you can confirm
everything installs and runs before you have real data.

## Using it with a real dataset

### Step 1 — Get the data
- Apply for DementiaBank / ADReSS access at https://dementia.talkbank.org/access
  (free, requires signing a data use agreement — do this early, approval takes time)
- OR use DementiaNet (no registration): https://github.com/shreyasgite/dementianet

### Step 2 — Organize it like this
```
data/
  transcripts/
    AD/   001.txt  002.txt  ...
    HC/   050.txt  051.txt  ...
  audio/
    AD/   001.wav  002.wav  ...
    HC/   050.wav  051.wav  ...
```
Filenames (without extension) should match across `transcripts/` and `audio/`
for the same speaker so the pipeline can pair them up. If you only have one
modality, just omit the other folder.

If your transcripts are in `.cha` (CHAT) format, as ADReSS/Pitt provide,
you can either convert them first:
```bash
pip install pylangacq --break-system-packages
python cha_to_txt.py --input_dir ./raw/AD --output_dir ./data/transcripts/AD
python cha_to_txt.py --input_dir ./raw/HC --output_dir ./data/transcripts/HC
```
or let the pipeline read `.cha` files directly when they are placed under
`data/transcripts/AD` and `data/transcripts/HC`.
If your audio is `.mp3` (as distributed by DementiaBank), convert to `.wav`
first, e.g. with ffmpeg: `ffmpeg -i in.mp3 out.wav`

### Step 3 — Run
```bash
python dementia_detection_pipeline.py --data_dir ./data
```
Options:
- `--no_audio` — text-only run
- `--no_text` — audio-only run
- `--out features.csv` — where to save the extracted feature matrix (useful
  for later re-use, or to plug into a different model/notebook)
- `--eval_mode holdout` — use a single train/test split instead of CV
- `--test_size 0.2` — size of the holdout test split

## What features are extracted

**Text (linguistic):**
type-token ratio & MATTR (lexical diversity), average word/sentence length,
filler word rate ("um", "uh"...), word repetition count, pronoun-to-noun
ratio and verb-to-noun ratio (word-finding difficulty proxies) — these are
standard markers used across the AD-detection literature.

**Audio (acoustic):**
pause ratio & count, speech rate proxy, pitch (F0) mean/std, 13 MFCCs
(mean+std), zero-crossing rate, spectral centroid.

Text and audio features are concatenated for the fused model — just leave
both `--no_text` and `--no_audio` unset.

**Text baseline:**
TF-IDF with n-grams is also evaluated directly on transcripts so you can
compare handcrafted linguistic biomarkers against a simpler bag-of-words
approach.

## Next steps for your thesis
1. Replace demo data with real ADReSS data and re-run — compare against
   published benchmark accuracy (~80–90% on ADReSS test set) to sanity-check
   your pipeline.
2. Try the ADReSSo variant (no manual transcripts provided) with an ASR
   system (e.g. OpenAI Whisper) in place of ground-truth transcripts, and
   compare accuracy to the manual-transcript condition — this is exactly the
   kind of experiment the earlier proposal outline suggested for a
   "robustness to ASR error" contribution.
3. Swap the hand-crafted features for pretrained embeddings (BERT for text,
   Wav2Vec2 for audio) as a stronger baseline / ablation comparison.
4. Use the *official* ADReSS train/test split (not just k-fold CV on the
   whole set) when reporting your final thesis numbers, so they're directly
   comparable to published results.

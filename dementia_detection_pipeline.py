"""
Dementia Detection from Speech and Text
=========================================
A full pipeline for binary classification (AD vs Healthy Control) using
linguistic features (from transcripts) and acoustic features (from audio),
individually and fused together. It also includes a lightweight TF-IDF
text baseline so the project can compare handcrafted features against a
simple bag-of-words approach without turning into an exact paper replica.

Works with any dataset organized as:

    data/
        transcripts/
            AD/   *.txt   or *.cha
            HC/   *.txt
        audio/
            AD/   *.wav, *.mp3, *.flac, *.m4a
            HC/   *.wav, *.mp3, *.flac, *.m4a

If you only have text OR only audio, the pipeline still runs (it just skips
the missing modality). This matches how ADReSS / Pitt Corpus / DementiaNet
are typically organized after you download and unzip them (rename their
folders to AD/HC if needed).

NOTE on .cha files (CHAT format used by DementiaBank/ADReSS):
    This script can read .cha files directly and will try to extract the
    participant's speech (PAR) through pylangacq when available. If you
    prefer to pre-convert, the accompanying `cha_to_txt.py` helper still
    works.

Usage
-----
    python dementia_detection_pipeline.py --data_dir ./data
    python dementia_detection_pipeline.py --demo        # runs on synthetic data

Author: generated for thesis prototyping. Please cite ADReSS/DementiaBank
appropriately if you use their data (see data use agreement).
"""

import argparse
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TEXT_EXTENSIONS = (".txt", ".cha")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")

# --------------------------------------------------------------------------
# Optional heavy imports are done lazily inside functions that need them,
# so the script can still run (e.g. --demo, text-only) even if librosa or
# nltk aren't fully set up yet.
# --------------------------------------------------------------------------


# ==========================================================================
# 1. TEXT FEATURE EXTRACTION (linguistic biomarkers)
# ==========================================================================

def extract_text_features(text: str) -> dict:
    """
    Extracts linguistic features known from the literature to correlate with
    AD: lexical diversity, syntactic complexity proxies, disfluencies,
    pronoun/noun ratios (a marker of word-finding difficulty), and idea
    density proxies.
    """
    import nltk
    for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
                "averaged_perceptron_tagger_eng"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

    from nltk import word_tokenize, pos_tag, sent_tokenize

    text = text.strip()
    if not text:
        return _empty_text_features()

    # Disfluency markers before cleaning (AD speech has more of these)
    filler_pattern = r"\b(uh|um|er|ah|hmm|you know|i mean)\b"
    fillers = re.findall(filler_pattern, text.lower())
    repetitions = len(re.findall(r"\b(\w+)\s+\1\b", text.lower()))  # "the the"

    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
    except Exception:
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        words = text.split()

    words_alpha = [w.lower() for w in words if w.isalpha()]
    n_words = len(words_alpha)
    n_sents = max(len(sentences), 1)

    if n_words == 0:
        return _empty_text_features()

    # Lexical diversity
    ttr = len(set(words_alpha)) / n_words                      # type-token ratio
    mattr = _moving_average_ttr(words_alpha, window=30)         # more robust for varying lengths

    # Word length / complexity
    avg_word_len = np.mean([len(w) for w in words_alpha])
    avg_sent_len = n_words / n_sents

    # POS-based features (noun/pronoun ratio drops in AD -> word-finding difficulty)
    try:
        tags = pos_tag(words)
        pos_counts = {}
        for _, tag in tags:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1
        n_nouns = sum(v for k, v in pos_counts.items() if k.startswith("NN"))
        n_pronouns = sum(v for k, v in pos_counts.items() if k.startswith("PRP"))
        n_verbs = sum(v for k, v in pos_counts.items() if k.startswith("VB"))
        pronoun_noun_ratio = n_pronouns / (n_nouns + 1e-6)
        verb_noun_ratio = n_verbs / (n_nouns + 1e-6)
    except Exception:
        pronoun_noun_ratio = 0.0
        verb_noun_ratio = 0.0

    return {
        "n_words": n_words,
        "n_sentences": n_sents,
        "avg_word_len": avg_word_len,
        "avg_sent_len": avg_sent_len,
        "ttr": ttr,
        "mattr": mattr,
        "filler_count": len(fillers),
        "filler_rate": len(fillers) / n_words,
        "repetition_count": repetitions,
        "pronoun_noun_ratio": pronoun_noun_ratio,
        "verb_noun_ratio": verb_noun_ratio,
    }


def _moving_average_ttr(words, window=30):
    """MATTR: more robust lexical diversity measure that doesn't shrink
    artificially with longer transcripts (a known TTR weakness)."""
    if len(words) < window:
        return len(set(words)) / max(len(words), 1)
    ratios = []
    for i in range(len(words) - window + 1):
        seg = words[i:i + window]
        ratios.append(len(set(seg)) / window)
    return float(np.mean(ratios))


def _empty_text_features():
    keys = ["n_words", "n_sentences", "avg_word_len", "avg_sent_len", "ttr",
            "mattr", "filler_count", "filler_rate", "repetition_count",
            "pronoun_noun_ratio", "verb_noun_ratio"]
    return {k: 0.0 for k in keys}


# ==========================================================================
# 2. AUDIO FEATURE EXTRACTION (acoustic / prosodic biomarkers)
# ==========================================================================

def extract_audio_features(audio_path: str) -> dict:
    """
    Extracts acoustic features known to correlate with AD: speech rate proxy,
    pause statistics (AD speakers pause more/longer), pitch variation, and
    MFCC statistics (spectral envelope, used broadly in speech pathology ML).
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(y) / sr
    if duration < 0.5:
        return _empty_audio_features()

    # --- Pause / silence analysis ---
    intervals = librosa.effects.split(y, top_db=30)  # non-silent intervals
    voiced_duration = sum((e - s) for s, e in intervals) / sr
    n_pauses = max(len(intervals) - 1, 0)
    pause_duration = duration - voiced_duration
    pause_ratio = pause_duration / duration
    speech_rate_proxy = voiced_duration / duration  # fraction of time actually speaking

    # --- Pitch (F0) ---
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        f0_voiced = f0[~np.isnan(f0)]
        f0_mean = float(np.mean(f0_voiced)) if len(f0_voiced) else 0.0
        f0_std = float(np.std(f0_voiced)) if len(f0_voiced) else 0.0
    except Exception:
        f0_mean, f0_std = 0.0, 0.0

    # --- MFCCs (13 coefficients, mean + std) ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    # --- Zero crossing rate & spectral centroid (voice quality proxies) ---
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    feats = {
        "duration_sec": duration,
        "n_pauses": n_pauses,
        "pause_ratio": pause_ratio,
        "speech_rate_proxy": speech_rate_proxy,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "zcr": zcr,
        "spectral_centroid": spec_centroid,
    }
    for i in range(13):
        feats[f"mfcc{i+1}_mean"] = float(mfcc_mean[i])
        feats[f"mfcc{i+1}_std"] = float(mfcc_std[i])
    return feats


def _empty_audio_features():
    keys = ["duration_sec", "n_pauses", "pause_ratio", "speech_rate_proxy",
             "f0_mean", "f0_std", "zcr", "spectral_centroid"]
    feats = {k: 0.0 for k in keys}
    for i in range(13):
        feats[f"mfcc{i+1}_mean"] = 0.0
        feats[f"mfcc{i+1}_std"] = 0.0
    return feats


# ==========================================================================
# 3. DATASET LOADING
# ==========================================================================


def _read_transcript_file(path: Path) -> str:
    """Read a transcript file, supporting both plain text and CHAT (.cha)."""
    if path.suffix.lower() == ".cha":
        try:
            import pylangacq

            reader = pylangacq.read_chat(str(path))
            try:
                words = reader.words(participants="PAR")
            except Exception:
                words = reader.words()
            return " ".join(words)
        except Exception:
            return path.read_text(errors="ignore")
    return path.read_text(errors="ignore")

def load_dataset(data_dir: str) -> pd.DataFrame:
    """
    Expects:
        data_dir/transcripts/AD/*.txt   data_dir/transcripts/HC/*.txt
        data_dir/audio/AD/*.wav         data_dir/audio/HC/*.wav
    Matches files by stem (filename without extension) across modalities.
    Either transcripts/ or audio/ may be absent -> that modality is skipped.
    """
    data_dir = Path(data_dir)
    text_dir = data_dir / "transcripts"
    audio_dir = data_dir / "audio"

    records = {}  # stem -> {"label":..., "text":..., "audio_path":...}

    if text_dir.exists():
        for label in ["AD", "HC"]:
            for ext in TEXT_EXTENSIONS:
                for f in (text_dir / label).glob(f"*{ext}"):
                    stem = f.stem
                    records.setdefault(stem, {"label": label})
                    records[stem]["text"] = _read_transcript_file(f)
                    records[stem]["label"] = label

    if audio_dir.exists():
        for label in ["AD", "HC"]:
            for ext in AUDIO_EXTENSIONS:
                for f in (audio_dir / label).glob(f"*{ext}"):
                    stem = f.stem
                    records.setdefault(stem, {"label": label})
                    records[stem]["audio_path"] = str(f)
                    records[stem]["label"] = label

    if not records:
        raise FileNotFoundError(
            f"No data found under {data_dir}. Expected transcripts/AD, "
            f"transcripts/HC, audio/AD, audio/HC subfolders."
        )

    rows = []
    for stem, r in records.items():
        rows.append({
            "id": stem,
            "label": r["label"],
            "text": r.get("text", ""),
            "audio_path": r.get("audio_path", None),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# 4. SYNTHETIC DEMO DATA (so you can test the pipeline before your real
#    dataset access / download is ready)
# ==========================================================================

def make_demo_data(n_per_class=25, seed=42):
    """Generates synthetic transcripts (no audio) to sanity-check the
    text pipeline end to end. Replace with real data ASAP -- this is only
    for verifying your code runs, NOT a substitute for a real dataset."""
    rng = np.random.default_rng(seed)
    hc_templates = [
        "The boy is reaching for a cookie jar while his sister watches him carefully.",
        "The mother is washing dishes at the sink and the water is overflowing onto the floor.",
        "There is a window above the sink showing a garden outside with trees.",
        "The little girl is asking her brother to give her a cookie from the jar.",
    ]
    ad_templates = [
        "The boy is uh reaching for the the cookie um thing there.",
        "She is washing um you know the the plates and uh water is uh going down.",
        "There's a a window there and uh outside is uh some trees I think.",
        "The girl she wants um the cookie from uh the boy up on the uh stool thing.",
    ]
    rows = []
    for i in range(n_per_class):
        rows.append({"id": f"hc_{i}", "label": "HC",
                     "text": " ".join(rng.choice(hc_templates, size=3)),
                     "audio_path": None})
        rows.append({"id": f"ad_{i}", "label": "AD",
                     "text": " ".join(rng.choice(ad_templates, size=3)),
                     "audio_path": None})
    return pd.DataFrame(rows)


# ==========================================================================
# 5. FEATURE MATRIX BUILDING
# ==========================================================================

def build_feature_matrix(df: pd.DataFrame, use_text=True, use_audio=True):
    text_feat_rows, audio_feat_rows = [], []
    for _, row in df.iterrows():
        if use_text:
            text_feat_rows.append(extract_text_features(row.get("text", "")))
        if use_audio and row.get("audio_path"):
            try:
                audio_feat_rows.append(extract_audio_features(row["audio_path"]))
            except Exception as e:
                print(f"  [warn] audio feature extraction failed for {row['id']}: {e}")
                audio_feat_rows.append(_empty_audio_features())
        elif use_audio:
            audio_feat_rows.append(_empty_audio_features())

    feats = pd.DataFrame()
    if use_text and text_feat_rows:
        feats = pd.concat([feats, pd.DataFrame(text_feat_rows).add_prefix("txt_")], axis=1)
    if use_audio and audio_feat_rows:
        feats = pd.concat([feats, pd.DataFrame(audio_feat_rows).add_prefix("aud_")], axis=1)

    feats["label"] = df["label"].values
    feats["id"] = df["id"].values
    return feats


# ==========================================================================
# 6. MODELING & EVALUATION
# ==========================================================================

def train_and_evaluate(feats: pd.DataFrame, label_col="label", id_cols=("id", "label")):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                  classification_report, confusion_matrix)

    X = feats.drop(columns=list(id_cols)).values
    y = LabelEncoder().fit_transform(feats[label_col].values)  # AD=0/HC=1 or similar

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    }

    n_splits = min(5, np.min(np.bincount(y)))  # can't have more folds than smallest class
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}
    for name, clf in models.items():
        k = min(20, X.shape[1])
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k)),
            ("clf", clf),
        ])
        y_pred = cross_val_predict(pipe, X, y, cv=cv)
        try:
            y_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, y_proba)
        except Exception:
            auc = float("nan")

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        results[name] = {"accuracy": acc, "f1": f1, "auc": auc}

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}  |  F1: {f1:.3f}  |  AUC: {auc:.3f}")
        print(classification_report(y, y_pred, target_names=["class_0", "class_1"]))
        print("Confusion matrix:\n", confusion_matrix(y, y_pred))

    return pd.DataFrame(results).T


def train_text_baselines(df: pd.DataFrame, text_col="text", label_col="label"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                  classification_report, confusion_matrix)

    texts = df[text_col].fillna("").astype(str).values
    y = LabelEncoder().fit_transform(df[label_col].values)

    models = {
        "TF-IDF + Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "TF-IDF + SVM (RBF)": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=1)),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced")),
        ]),
    }

    n_splits = min(5, np.min(np.bincount(y)))
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}
    for name, pipe in models.items():
        y_pred = cross_val_predict(pipe, texts, y, cv=cv)
        try:
            y_proba = cross_val_predict(pipe, texts, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, y_proba)
        except Exception:
            auc = float("nan")

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        results[name] = {"accuracy": acc, "f1": f1, "auc": auc}

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}  |  F1: {f1:.3f}  |  AUC: {auc:.3f}")
        print(classification_report(y, y_pred, target_names=["class_0", "class_1"]))
        print("Confusion matrix:\n", confusion_matrix(y, y_pred))

    return pd.DataFrame(results).T


def train_and_evaluate_holdout(feats: pd.DataFrame, label_col="label", id_cols=("id", "label"), test_size=0.2):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                  classification_report, confusion_matrix)

    X = feats.drop(columns=list(id_cols)).values
    y = LabelEncoder().fit_transform(feats[label_col].values)

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least two classes to train a classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    }

    results = {}
    for name, clf in models.items():
        k = min(20, X_train.shape[1])
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k)),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = float("nan")

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results[name] = {"accuracy": acc, "f1": f1, "auc": auc}

        print(f"\n=== {name} (holdout) ===")
        print(f"Accuracy: {acc:.3f}  |  F1: {f1:.3f}  |  AUC: {auc:.3f}")
        print(classification_report(y_test, y_pred, target_names=["class_0", "class_1"]))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    return pd.DataFrame(results).T


# ==========================================================================
# 7. MAIN
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="Dementia detection from speech and text")
    parser.add_argument("--data_dir", type=str, default=None,
                         help="Path to data/ folder with transcripts/ and audio/ subfolders")
    parser.add_argument("--demo", action="store_true",
                         help="Run on synthetic demo data (text-only) to verify pipeline works")
    parser.add_argument("--no_audio", action="store_true", help="Skip audio features")
    parser.add_argument("--no_text", action="store_true", help="Skip text features")
    parser.add_argument("--out", type=str, default="features.csv",
                         help="Where to save the extracted feature matrix")
    parser.add_argument("--eval_mode", type=str, default="cv", choices=["cv", "holdout"],
                         help="Evaluation strategy: cross-validation or holdout split")
    parser.add_argument("--test_size", type=float, default=0.2,
                         help="Holdout test size when --eval_mode holdout is used")
    args = parser.parse_args()

    if args.demo:
        print("Running in DEMO mode with synthetic transcripts (no real audio).")
        df = make_demo_data()
        use_audio = False
    else:
        if not args.data_dir:
            raise SystemExit("Provide --data_dir or use --demo to test the pipeline.")
        df = load_dataset(args.data_dir)
        use_audio = not args.no_audio

    print(f"Loaded {len(df)} samples: {df['label'].value_counts().to_dict()}")

    print("Extracting features (this can take a while for audio)...")
    feats = build_feature_matrix(df, use_text=not args.no_text, use_audio=use_audio)
    feats.to_csv(args.out, index=False)
    print(f"Saved feature matrix -> {args.out}  (shape={feats.shape})")

    if args.eval_mode == "holdout":
        print("\nTraining & evaluating handcrafted-feature models (holdout split)...")
        results = train_and_evaluate_holdout(feats, test_size=args.test_size)
    else:
        print("\nTraining & evaluating handcrafted-feature models (5-fold stratified CV)...")
        results = train_and_evaluate(feats)
    print("\n=== SUMMARY ===")
    print(results)

    if not args.no_text and "text" in df.columns:
        print("\nTraining & evaluating TF-IDF text baselines (5-fold stratified CV)...")
        text_results = train_text_baselines(df)
        print("\n=== TEXT BASELINE SUMMARY ===")
        print(text_results)


if __name__ == "__main__":
    main()

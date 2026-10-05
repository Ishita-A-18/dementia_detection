"""
Cross-Corpus Evaluation
Train on one corpus, test on another to measure generalization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report


def extract_text_features(text: str) -> dict:
    """Minimal feature extraction (same as main pipeline)."""
    from nltk import word_tokenize, pos_tag, sent_tokenize
    
    text = text.strip()
    if not text:
        return {
            "n_words": 0, "n_sentences": 0, "avg_word_len": 0, "avg_sent_len": 0,
            "ttr": 0, "mattr": 0, "filler_count": 0, "filler_rate": 0,
            "repetition_count": 0, "pronoun_noun_ratio": 0, "verb_noun_ratio": 0,
        }
    
    filler_pattern = r"\b(uh|um|er|ah|hmm|you know|i mean)\b"
    fillers = re.findall(filler_pattern, text.lower())
    repetitions = len(re.findall(r"\b(\w+)\s+\1\b", text.lower()))
    
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
    except:
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        words = text.split()
    
    words_alpha = [w.lower() for w in words if w.isalpha()]
    n_words = len(words_alpha)
    n_sents = max(len(sentences), 1)
    
    if n_words == 0:
        return {"n_words": 0, "n_sentences": 0, "avg_word_len": 0, "avg_sent_len": 0,
                "ttr": 0, "mattr": 0, "filler_count": 0, "filler_rate": 0,
                "repetition_count": 0, "pronoun_noun_ratio": 0, "verb_noun_ratio": 0}
    
    ttr = len(set(words_alpha)) / n_words
    mattr = _moving_average_ttr(words_alpha, window=30)
    avg_word_len = np.mean([len(w) for w in words_alpha])
    avg_sent_len = n_words / n_sents
    
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
    except:
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
    if len(words) < window:
        return len(set(words)) / max(len(words), 1)
    ratios = []
    for i in range(len(words) - window + 1):
        seg = words[i:i + window]
        ratios.append(len(set(seg)) / window)
    return float(np.mean(ratios)) if ratios else 0.0


def load_corpus_data(corpus_folder):
    """Load data from a corpus folder (Pitt, DePaul, etc.)."""
    corpus_path = Path(corpus_folder)
    records = {}
    
    # Try to find AD/HC or Dementia/Control folders
    ad_candidates = ["AD", "Dementia", corpus_path / "Dementia" / "*"]
    hc_candidates = ["HC", "Control", corpus_path / "Control" / "*"]
    
    for label_name in ["Dementia", "AD"]:
        label_path = corpus_path / label_name
        if label_path.exists():
            for f in label_path.rglob("*.cha"):
                try:
                    import pylangacq
                    reader = pylangacq.read_chat(str(f))
                    try:
                        words = reader.words(participants="PAR")
                    except:
                        words = reader.words()
                    text = " ".join(words)
                except:
                    text = f.read_text(errors="ignore")
                
                records[f.stem] = {"label": "AD", "text": text}
    
    for label_name in ["Control", "HC"]:
        label_path = corpus_path / label_name
        if label_path.exists():
            for f in label_path.rglob("*.cha"):
                try:
                    import pylangacq
                    reader = pylangacq.read_chat(str(f))
                    try:
                        words = reader.words(participants="PAR")
                    except:
                        words = reader.words()
                    text = " ".join(words)
                except:
                    text = f.read_text(errors="ignore")
                
                records[f.stem] = {"label": "HC", "text": text}
    
    if not records:
        return None
    
    return pd.DataFrame([{"id": k, **v} for k, v in records.items()])


def cross_corpus_eval(train_corpus, test_corpus):
    """Train on one corpus, test on another."""
    print(f"\n{'='*70}")
    print(f"Cross-Corpus: Train on {train_corpus.stem}, Test on {test_corpus.stem}")
    print(f"{'='*70}")
    
    train_df = load_corpus_data(train_corpus)
    test_df = load_corpus_data(test_corpus)
    
    if train_df is None or test_df is None:
        print(f"[!] Could not load data from {train_corpus.stem} or {test_corpus.stem}")
        return None
    
    print(f"Train set: {len(train_df)} samples ({train_df['label'].value_counts().to_dict()})")
    print(f"Test set: {len(test_df)} samples ({test_df['label'].value_counts().to_dict()})")
    
    # Extract features
    train_feats = [extract_text_features(text) for text in train_df["text"]]
    test_feats = [extract_text_features(text) for text in test_df["text"]]
    
    X_train = pd.DataFrame(train_feats).values
    X_test = pd.DataFrame(test_feats).values
    y_train = LabelEncoder().fit_transform(train_df["label"].values)
    y_test = LabelEncoder().fit_transform(test_df["label"].values)
    
    # Train model
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=min(20, X_train.shape[1]))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipe.predict(X_test)
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = float("nan")
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    print(f"  AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["AD", "HC"]))
    
    return {"train": train_corpus.stem, "test": test_corpus.stem, "accuracy": acc, "f1": f1, "auc": auc}


def main():
    corpora = [Path(d) for d in ["Pitt", "DePaul", "Baycrest"] if Path(d).exists()]
    
    if len(corpora) < 2:
        print("[!] Need at least 2 corpus folders (Pitt, DePaul, etc.)")
        return
    
    print("[*] Cross-Corpus Generalization Study")
    results = []
    
    for i, train_corp in enumerate(corpora):
        for test_corp in corpora:
            if train_corp != test_corp:
                result = cross_corpus_eval(train_corp, test_corp)
                if result:
                    results.append(result)
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("cross_corpus_results.csv", index=False)
        print(f"\n{'='*70}")
        print("All Cross-Corpus Results:")
        print(results_df.to_string(index=False))
        print(f"\nResults saved -> cross_corpus_results.csv")


if __name__ == "__main__":
    main()

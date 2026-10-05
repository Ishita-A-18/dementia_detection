"""
Embedding-based baselines for dementia detection.
Compares handcrafted features against pretrained deep learning embeddings.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def _speaker_id_from_sample_id(sample_id):
    if not isinstance(sample_id, str) or not sample_id:
        return str(sample_id)
    for sep in ("-", "_", "/", "\\"):
        if sep in sample_id:
            return sample_id.split(sep)[0]
    return sample_id

def extract_bert_embeddings(texts):
    """Extract sentence embeddings using sentence-transformers (BERT-based)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[!] sentence-transformers not installed. Install with:")
        print("    pip install sentence-transformers")
        return None
    
    print("Loading BERT model (this may take a moment on first run)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight, fast
    print("Extracting embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def train_embedding_baseline(embedding_matrix, labels, groups, model_type="logistic"):
    """Train classifier on embedding features."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
    
    y = LabelEncoder().fit_transform(labels)

    speaker_df = pd.DataFrame({"speaker_id": groups, "label": y}).drop_duplicates("speaker_id")
    try:
        train_speakers, test_speakers = train_test_split(
            speaker_df["speaker_id"],
            test_size=0.2,
            random_state=42,
            stratify=speaker_df["label"],
        )
    except ValueError:
        train_speakers, test_speakers = train_test_split(
            speaker_df["speaker_id"],
            test_size=0.2,
            random_state=42,
        )

    train_mask = np.isin(groups, train_speakers)
    test_mask = np.isin(groups, test_speakers)
    X_train, X_test = embedding_matrix[train_mask], embedding_matrix[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    models = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "svm": SVC(kernel="rbf", probability=True, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    }
    
    clf = models.get(model_type, models["logistic"])
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")
    
    print(f"\n=== {model_type.upper()} on BERT Embeddings (holdout test) ===")
    print(f"Accuracy: {acc:.3f}  |  F1: {f1:.3f}  |  AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["AD", "HC"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    
    return {"accuracy": acc, "f1": f1, "auc": auc}


def main():
    parser = argparse.ArgumentParser(description="Embedding-based dementia detection baselines")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to data/ folder with transcripts/")
    parser.add_argument("--use_pretrained", action="store_true",
                        help="Use pretrained BERT embeddings (requires sentence-transformers)")
    args = parser.parse_args()
    
    # Load dataset
    data_dir = Path(args.data_dir)
    records = {}
    
    for label in ["AD", "HC"]:
        for f in list((data_dir / "transcripts" / label).glob("*.cha")) + list((data_dir / "transcripts" / label).glob("*.txt")):
            stem = f.stem
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
            
            records[stem] = {"label": label, "text": text, "speaker_id": _speaker_id_from_sample_id(stem)}
    
    if not records:
        raise FileNotFoundError(f"No transcripts found in {data_dir}")
    
    df = pd.DataFrame([{"id": k, **v} for k, v in records.items()])
    print(f"Loaded {len(df)} samples: {df['label'].value_counts().to_dict()}")
    
    if args.use_pretrained:
        # Use BERT embeddings
        embeddings = extract_bert_embeddings(df["text"].values)
        if embeddings is not None:
            print(f"\nEmbedding shape: {embeddings.shape}")
            print("=" * 60)
            results = {}
            groups = df["speaker_id"].astype(str).values
            for model_type in ["logistic", "svm", "rf"]:
                results[model_type] = train_embedding_baseline(embeddings, df["label"].values, groups, model_type)
            
            summary = pd.DataFrame(results).T
            print("\n=== EMBEDDING-BASED SUMMARY ===")
            print(summary)
            summary.to_csv("embeddings_baseline_results.csv")
            print("Results saved -> embeddings_baseline_results.csv")
    else:
        print("[*] Run with --use_pretrained to test BERT embeddings")
        print("    python embeddings_baseline.py --data_dir ./data --use_pretrained")


if __name__ == "__main__":
    main()

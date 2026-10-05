"""
Feature Ablation Study for Dementia Detection
Identifies which linguistic features are most predictive of AD.
Useful for thesis methods section.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score


def _speaker_id_from_sample_id(sample_id):
    if not isinstance(sample_id, str) or not sample_id:
        return str(sample_id)
    for sep in ("-", "_", "/", "\\"):
        if sep in sample_id:
            return sample_id.split(sep)[0]
    return sample_id


def run_ablation_study(features_csv="features.csv", test_size=0.2):
    """
    Train model with each feature removed one at a time.
    Measures accuracy drop to determine feature importance.
    """
    feats = pd.read_csv(features_csv)
    
    # Drop id/label metadata columns
    drop_cols = [c for c in ["id", "label", "speaker_id"] if c in feats.columns]
    X_full = feats.drop(columns=drop_cols).values
    y = LabelEncoder().fit_transform(feats["label"].values)
    
    feature_names = feats.drop(columns=drop_cols).columns.tolist()

    if "speaker_id" in feats.columns:
        speaker_ids = feats["speaker_id"].astype(str).values
    else:
        speaker_ids = feats["id"].astype(str).apply(_speaker_id_from_sample_id).values

    speaker_df = pd.DataFrame({"speaker_id": speaker_ids, "label": y}).drop_duplicates("speaker_id")
    try:
        train_speakers, test_speakers = train_test_split(
            speaker_df["speaker_id"],
            test_size=test_size,
            random_state=42,
            stratify=speaker_df["label"],
        )
    except ValueError:
        train_speakers, test_speakers = train_test_split(
            speaker_df["speaker_id"],
            test_size=test_size,
            random_state=42,
        )

    train_mask = np.isin(speaker_ids, train_speakers)
    test_mask = np.isin(speaker_ids, test_speakers)
    
    # Train with all features (baseline)
    X_train, X_test, y_train, y_test = X_full[train_mask], X_full[test_mask], y[train_mask], y[test_mask]
    
    baseline_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=min(20, X_train.shape[1]))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    baseline_pipe.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline_pipe.predict(X_test))
    
    print(f"\nBaseline Accuracy (all features): {baseline_acc:.4f}")
    print("=" * 70)
    print(f"{'Feature':<30} {'Accuracy':<12} {'Degradation':<12} {'% Impact':<10}")
    print("=" * 70)
    
    ablation_results = []
    
    # Test with each feature removed
    for feature_idx, feature_name in enumerate(feature_names):
        # Create feature set without this feature
        X_full_ablated = np.delete(X_full, feature_idx, axis=1)
        X_train_ab, X_test_ab = X_full_ablated[train_mask], X_full_ablated[test_mask]
        y_train_ab, y_test_ab = y[train_mask], y[test_mask]
        
        pipe_ablated = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=min(20, X_train_ab.shape[1]))),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        pipe_ablated.fit(X_train_ab, y_train_ab)
        ablated_acc = accuracy_score(y_test_ab, pipe_ablated.predict(X_test_ab))
        
        degradation = baseline_acc - ablated_acc
        pct_impact = (degradation / baseline_acc) * 100 if baseline_acc > 0 else 0
        
        status = "↑ IMPORTANT" if degradation > 0.02 else ""
        
        print(f"{feature_name:<30} {ablated_acc:<12.4f} {degradation:<12.4f} {pct_impact:<10.1f} {status}")
        
        ablation_results.append({
            "feature": feature_name,
            "accuracy_without": ablated_acc,
            "degradation": degradation,
            "pct_impact": pct_impact,
        })
    
    print("=" * 70)
    
    # Save results
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df = ablation_df.sort_values("degradation", ascending=False)
    ablation_df.to_csv("feature_ablation_results.csv", index=False)
    
    print(f"\nResults saved -> feature_ablation_results.csv")
    print(f"\nTop 5 Most Important Features:")
    for i, (idx, row) in enumerate(ablation_df.head().iterrows(), 1):
        print(f"  {i}. {row['feature']}: {row['degradation']:.4f} accuracy drop ({row['pct_impact']:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="features.csv")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    
    if not Path(args.features).exists():
        print(f"[!] {args.features} not found. Run the main pipeline first:")
        print("    python dementia_detection_pipeline.py --data_dir ./data --no_audio")
    else:
        print("\n[*] Feature Ablation Study")
        print("[*] Removing each feature and measuring accuracy impact...")
        run_ablation_study(args.features, args.test_size)

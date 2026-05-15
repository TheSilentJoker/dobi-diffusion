from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from breastdiffusion.image_features import extract_feature_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated DOBI NIR images against real images.")
    parser.add_argument("--real-manifest", default="data/manifests/dobi_prompts.csv")
    parser.add_argument("--generated-manifest", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="reports/generation_eval.json")
    return parser.parse_args()


def _frechet_distance(real: np.ndarray, fake: np.ndarray) -> float:
    mu1, mu2 = real.mean(axis=0), fake.mean(axis=0)
    sigma1, sigma2 = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def _kid(real: np.ndarray, fake: np.ndarray) -> float:
    gamma = 1.0 / real.shape[1]
    k_xx = (gamma * real @ real.T + 1.0) ** 3
    k_yy = (gamma * fake @ fake.T + 1.0) ** 3
    k_xy = (gamma * real @ fake.T + 1.0) ** 3
    m, n = len(real), len(fake)
    xx = (k_xx.sum() - np.trace(k_xx)) / max(m * (m - 1), 1)
    yy = (k_yy.sum() - np.trace(k_yy)) / max(n * (n - 1), 1)
    xy = k_xy.mean()
    return float(xx + yy - 2.0 * xy)


def main() -> None:
    args = parse_args()
    real_manifest = pd.read_csv(args.real_manifest)
    generated_manifest = pd.read_csv(args.generated_manifest)
    real_manifest = real_manifest[real_manifest["split"] == args.split].copy()

    reports: dict[str, dict[str, float]] = {}
    for label in sorted(set(real_manifest["label"]).intersection(set(generated_manifest["label"]))):
        real_df = real_manifest[real_manifest["label"] == label]
        fake_df = generated_manifest[generated_manifest["label"] == label]
        if len(real_df) < 2 or len(fake_df) < 2:
            continue

        real_features = extract_feature_matrix(real_df["image_path"].astype(str).tolist())
        fake_features = extract_feature_matrix(fake_df["image_path"].astype(str).tolist())
        scaler = StandardScaler().fit(real_features)
        real_scaled = scaler.transform(real_features)
        fake_scaled = scaler.transform(fake_features)
        n_components = min(64, real_scaled.shape[0] - 1, fake_scaled.shape[0] - 1, real_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42).fit(real_scaled)
        real_emb = pca.transform(real_scaled)
        fake_emb = pca.transform(fake_scaled)
        nearest_similarity = cosine_similarity(fake_emb, real_emb).max(axis=1)

        reports[str(label)] = {
            "real_count": float(len(real_df)),
            "generated_count": float(len(fake_df)),
            "feature_fid": _frechet_distance(real_emb, fake_emb),
            "feature_kid": _kid(real_emb, fake_emb),
            "mean_nearest_real_cosine": float(nearest_similarity.mean()),
            "max_nearest_real_cosine": float(nearest_similarity.max()),
            "real_feature_mean": float(real_features.mean()),
            "generated_feature_mean": float(fake_features.mean()),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

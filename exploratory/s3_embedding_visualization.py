import argparse
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

from sequence_loading import BUCKET, OUTPUT_PREFIX, load_embedding_batch_from_s3


def compute_pca(df: pd.DataFrame, emb_prefix: str = "emb_", pca_dim: int = 2):
    emb_cols = [c for c in df.columns if c.startswith(emb_prefix)]
    if len(emb_cols) == 0:
        raise ValueError("No embedding columns found in the S3 parquet file.")

    X = df[emb_cols].to_numpy(dtype=float)
    pca = PCA(n_components=pca_dim)
    X_pca = pca.fit_transform(X)

    df_out = df.copy()
    for i in range(pca_dim):
        df_out[f"pc_{i+1}"] = X_pca[:, i]

    return df_out, X_pca, pca


def plot_pca(
    X_pca,
    pca_dim: int = 2,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    color: Optional[pd.Series] = None,
):
    if pca_dim == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=color,
            cmap="tab10",
            s=45,
            alpha=0.75,
            edgecolors="w",
            linewidth=0.4,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title or "Embeddings PCA (2D)")
        if color is not None:
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label("color")
    elif pca_dim == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            X_pca[:, 2],
            c=color,
            cmap="tab10",
            s=45,
            alpha=0.75,
            edgecolors="w",
            linewidth=0.4,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(title or "Embeddings PCA (3D)")
        if color is not None:
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label("color")
    else:
        raise ValueError("pca_dim must be 2 or 3")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
        print(f"Saved PCA figure to {save_path}")
    plt.show()


def load_and_plot(
    batch_num: int = 1,
    pca_dim: int = 2,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    emb_prefix: str = "emb_",
    color_col: Optional[str] = None,
    save_path: Optional[str] = None,
):
    bucket = bucket or BUCKET
    prefix = prefix or OUTPUT_PREFIX
    df = load_embedding_batch_from_s3(batch_num=batch_num, bucket=bucket, prefix=prefix)
    df_out, X_pca, pca = compute_pca(df, emb_prefix=emb_prefix, pca_dim=pca_dim)

    print(f"Loaded {len(df_out)} rows with {len([c for c in df_out.columns if c.startswith(emb_prefix)])} embedding columns")
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)

    color = None
    if color_col is not None:
        if color_col not in df_out.columns:
            raise ValueError(f"Color column '{color_col}' not found in DataFrame")
        color = df_out[color_col]

    plot_pca(
        X_pca,
        pca_dim=pca_dim,
        title=f"S3 embeddings PCA batch {batch_num}",
        save_path=save_path,
        color=color,
    )
    return df_out, X_pca, pca


def parse_args():
    parser = argparse.ArgumentParser(description="Load S3 embeddings and plot PCA")
    parser.add_argument("--batch-num", type=int, default=1, help="S3 embedding batch number")
    parser.add_argument("--pca-dim", type=int, choices=[2, 3], default=2, help="PCA dimension")
    parser.add_argument("--bucket", type=str, default=None, help="S3 bucket name (uses BUCKET from sequence_loading if omitted)")
    parser.add_argument("--prefix", type=str, default=None, help="S3 prefix for embedding files (uses OUTPUT_PREFIX from sequence_loading if omitted)")
    parser.add_argument("--emb-prefix", type=str, default="emb_", help="Embedding column prefix")
    parser.add_argument("--color-col", type=str, default=None, help="Optional DataFrame column to use for coloring points")
    parser.add_argument("--save-path", type=str, default=None, help="Optional path to save PCA figure")
    return parser.parse_args()


def main():
    args = parse_args()
    load_and_plot(
        batch_num=args.batch_num,
        pca_dim=args.pca_dim,
        bucket=args.bucket,
        prefix=args.prefix,
        emb_prefix=args.emb_prefix,
        color_col=args.color_col,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()

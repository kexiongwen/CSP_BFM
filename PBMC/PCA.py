"""Build the fixed ``X_scaled`` PCA reference used in ``PBMC/plan.md``."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __package__:
    from .data_processing import (
        PreprocessConfig,
        PreprocessedPBMC,
        _jsonable,
        _software_versions,
        preprocess_pbmc3k,
    )
    from .downstream import DownstreamConfig, run_downstream, write_top_markers
else:
    from data_processing import (
        PreprocessConfig,
        PreprocessedPBMC,
        _jsonable,
        _software_versions,
        preprocess_pbmc3k,
    )
    from downstream import DownstreamConfig, run_downstream, write_top_markers

__all__ = [
    "PCAConfig",
    "run_pca_scaled",
    "run_pca_baseline",
    "save_pca_baseline",
    "save_baseline_figures",
    "save_top_markers",
]


@dataclass(frozen=True)
class PCAConfig:
    n_comps: int = 50
    n_pcs: int = 40
    n_neighbors: int = 10
    leiden_resolution: float = 0.7
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.n_comps < 1:
            raise ValueError("n_comps must be positive")
        if not 1 <= self.n_pcs <= self.n_comps:
            raise ValueError("n_pcs must be in [1, n_comps]")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        if self.leiden_resolution <= 0:
            raise ValueError("leiden_resolution must be positive")
        if self.random_state < 0:
            raise ValueError("random_state must be nonnegative")


def run_pca_scaled(
    prepared: PreprocessedPBMC,
    *,
    config: PCAConfig | None = None,
    preprocess_config: PreprocessConfig | None = None,
) -> ad.AnnData:
    config = config or PCAConfig()
    adata = prepared.adata
    model = adata[:, prepared.hvg_names].copy()
    model.X = prepared.X_scaled.copy()
    sc.pp.pca(
        model,
        n_comps=config.n_comps,
        zero_center=True,
        svd_solver="arpack",
        random_state=config.random_state,
        mask_var=None,
        dtype="float32",
    )
    adata.obsm["X_pca_scaled"] = np.asarray(model.obsm["X_pca"]).copy()
    adata.uns["pca_scaled"] = {
        "input_matrix": "X_scaled",
        "params_json": json.dumps(_jsonable(model.uns["pca"]["params"])),
        "variance": np.asarray(model.uns["pca"]["variance"]).copy(),
        "variance_ratio": np.asarray(model.uns["pca"]["variance_ratio"]).copy(),
        "loadings": np.asarray(model.varm["PCs"]).copy(),
        "hvg_names": prepared.hvg_names.to_numpy(dtype=str),
    }
    run = run_downstream(
        adata,
        representation="X_pca_scaled",
        name="scaled",
        config=DownstreamConfig(
            n_pcs=config.n_pcs,
            n_neighbors=config.n_neighbors,
            leiden_resolution=config.leiden_resolution,
            random_state=config.random_state,
        ),
    )
    adata.uns["pca_baseline"] = {
        "config_json": json.dumps(_jsonable(asdict(config)), sort_keys=True),
        "preprocess_config_json": json.dumps(
            _jsonable(asdict(preprocess_config)) if preprocess_config is not None else None
        ),
        "software": _software_versions(),
        "runs": {"scaled": run},
    }
    return adata


def run_pca_baseline(
    data_path: str | Path,
    *,
    config: PCAConfig | None = None,
    preprocess_config: PreprocessConfig | None = None,
) -> ad.AnnData:
    preprocess_config = preprocess_config or PreprocessConfig()
    prepared = preprocess_pbmc3k(data_path, config=preprocess_config)
    return run_pca_scaled(prepared, config=config, preprocess_config=preprocess_config)


def save_pca_baseline(adata: ad.AnnData, output: str | Path) -> Path:
    if "pca_baseline" not in adata.uns:
        raise ValueError("run_pca_scaled must be called before saving")
    output = Path(output)
    if output.suffix.lower() != ".h5ad":
        raise ValueError(f"output must end in .h5ad, got {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output, compression="gzip")
    return output


def save_baseline_figures(adata: ad.AnnData, output_dir: str | Path) -> list[Path]:
    if "pca_baseline" not in adata.uns:
        raise ValueError("run_pca_scaled must be called before plotting")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ratio = np.asarray(adata.uns["pca_scaled"]["variance_ratio"])
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(np.arange(1, len(ratio) + 1), ratio, marker=".", linewidth=1)
    ax.set_xlabel("principal component")
    ax.set_ylabel("variance ratio")
    ax.set_title("PCA variance ratio: scaled")
    variance_path = output_dir / "pca_variance_scaled.png"
    fig.savefig(variance_path, dpi=150)
    plt.close(fig)
    run = adata.uns["pca_baseline"]["runs"]["scaled"]
    fig = sc.pl.embedding(
        adata, basis="umap_scaled", color=run["leiden"], show=False, return_fig=True
    )
    umap_path = output_dir / "umap_scaled.png"
    fig.savefig(umap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [variance_path, umap_path]


def save_top_markers(
    adata: ad.AnnData, output_prefix: str | Path, n: int = 5
) -> tuple[list[dict], Path, Path]:
    if "pca_baseline" not in adata.uns:
        raise ValueError("run_pca_scaled must be called before saving markers")
    marker_key = adata.uns["pca_baseline"]["runs"]["scaled"]["markers"]
    return write_top_markers(adata, marker_key, "PCA_scaled", output_prefix, n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_path", nargs="?", type=Path,
        default=Path("PBMC/data/pbmc3k_raw.h5ad"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("PBMC/data/notebook_scaled_pca_baseline.h5ad"),
    )
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    adata = run_pca_baseline(args.data_path)
    output = save_pca_baseline(adata, args.output)
    counts = adata.obs["leiden_scaled"].value_counts().sort_index()
    print(f"scaled: {len(counts)} Leiden clusters; sizes={counts.to_dict()}")
    print(f"Saved PCA baseline to {output}")
    if args.figure_dir is not None:
        paths = save_baseline_figures(adata, args.figure_dir)
        print(f"Saved {len(paths)} figures to {args.figure_dir}")
    _, csv_path, md_path = save_top_markers(
        adata, output.with_name(f"{output.stem}_scaled_top5_markers")
    )
    print(f"Saved scaled top-5 markers to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()

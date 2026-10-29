"""Shared neighbor-graph, embedding, clustering, and marker analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

__all__ = ["DownstreamConfig", "require_igraph", "run_downstream", "write_top_markers"]


@dataclass(frozen=True)
class DownstreamConfig:
    n_pcs: int = 50
    n_neighbors: int = 10
    leiden_resolution: float = 0.4
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.n_pcs < 1:
            raise ValueError("n_pcs must be positive")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        if self.leiden_resolution <= 0:
            raise ValueError("leiden_resolution must be positive")
        if self.random_state < 0:
            raise ValueError("random_state must be nonnegative")


def require_igraph() -> None:
    try:
        import igraph  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "the tutorial Leiden flavor='igraph' requires python-igraph. "
            "Install it in the PBMC analysis environment, for example with "
            "`python -m pip install python-igraph`."
        ) from exc


def run_downstream(
    adata: ad.AnnData,
    *,
    representation: str,
    name: str,
    config: DownstreamConfig,
) -> dict[str, str]:
    require_igraph()
    if representation not in adata.obsm:
        raise KeyError(f"missing representation adata.obsm[{representation!r}]")
    if adata.obsm[representation].shape[1] < config.n_pcs:
        raise ValueError(
            f"{representation} has {adata.obsm[representation].shape[1]} columns, "
            f"but n_pcs={config.n_pcs}"
        )
    scores = adata.obsm[representation][:, :config.n_pcs]
    if not np.isfinite(scores).all():
        raise FloatingPointError(f"{representation} contains non-finite scores")
    if np.any(np.std(scores, axis=0) == 0):
        raise ValueError(f"{representation} contains degenerate score columns")

    neighbors_key = f"neighbors_{name}"
    umap_key = f"umap_{name}"
    cluster_key = f"leiden_{name}"
    marker_key = f"markers_{name}"
    sc.pp.neighbors(
        adata,
        n_neighbors=config.n_neighbors,
        n_pcs=config.n_pcs,
        use_rep=representation,
        metric="euclidean",
        random_state=config.random_state,
        key_added=neighbors_key,
    )
    sc.tl.umap(
        adata,
        neighbors_key=neighbors_key,
        random_state=config.random_state,
        key_added=umap_key,
    )
    sc.tl.leiden(
        adata,
        resolution=config.leiden_resolution,
        random_state=config.random_state,
        flavor="igraph",
        n_iterations=2,
        directed=False,
        neighbors_key=neighbors_key,
        key_added=cluster_key,
    )
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        mask_var=None,
        use_raw=False,
        method="wilcoxon",
        corr_method="benjamini-hochberg",
        key_added=marker_key,
    )
    return {
        "representation": representation,
        "neighbors": neighbors_key,
        "umap": umap_key,
        "leiden": cluster_key,
        "markers": marker_key,
    }


def write_top_markers(
    adata: ad.AnnData,
    marker_key: str,
    method: str,
    output_prefix: str | Path,
    n: int = 5,
) -> tuple[list[dict], Path, Path]:
    if n < 1:
        raise ValueError("n must be positive")
    if marker_key not in adata.uns:
        raise KeyError(f"missing marker result {marker_key!r}")
    result = adata.uns[marker_key]
    required = {"names", "scores", "logfoldchanges", "pvals", "pvals_adj"}
    missing = required.difference(result)
    if missing:
        raise KeyError(f"marker result lacks fields: {sorted(missing)}")

    rows = []
    for group in result["names"].dtype.names:
        if len(result["names"][group]) < n:
            raise ValueError(f"cluster {group} has fewer than {n} ranked genes")
        for rank in range(n):
            rows.append({
                "method": method,
                "cluster": str(group),
                "rank": rank + 1,
                "gene": str(result["names"][group][rank]),
                "score": float(result["scores"][group][rank]),
                "logfoldchange": float(result["logfoldchanges"][group][rank]),
                "pval": float(result["pvals"][group][rank]),
                "pval_adj": float(result["pvals_adj"][group][rank]),
            })

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["method"], row["cluster"]), []).append(row["gene"])
    lines = [
        f"# Top {n} markers for {method}",
        "",
        "| Method | Cluster | Markers |",
        "|---|---|---|",
    ]
    for (method_name, cluster), genes in grouped.items():
        lines.append(f"| {method_name} | {cluster} | {', '.join(genes)} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows, csv_path, md_path

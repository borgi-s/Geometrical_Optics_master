# src/dfxm_geo/scoring/identify.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .engine import score_matrix, score_target
from .target import resample_to_grid
from .types import CandidateLabel, CandidateLibrary


@dataclass(frozen=True)
class RankedMatch:
    score: float
    label: CandidateLabel
    scan_index: int
    source_file: str


@dataclass
class IdentifiabilityResult:
    matrix: np.ndarray
    labels: list[CandidateLabel]
    top1_accuracy: float
    per_class_accuracy: dict
    confusion: np.ndarray
    class_order: list

    def save(self, out_dir: str | Path, *, plots: bool = False) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "matrix.npy", self.matrix)
        np.save(out / "confusion.npy", self.confusion)
        labels_json = [
            {
                "slip_plane_normal": list(lbl.slip_plane_normal),
                "burgers": list(lbl.burgers),
                "rotation_deg": lbl.rotation_deg,
                "gb_cos": lbl.gb_cos,
                "gb_visible": lbl.gb_visible,
                "q_hkl": list(lbl.q_hkl),
                "scan_index": lbl.scan_index,
                "source_file": lbl.source_file,
            }
            for lbl in self.labels
        ]
        (out / "labels.json").write_text(json.dumps(labels_json, indent=2))
        metrics = {
            "top1_accuracy": self.top1_accuracy,
            "per_class_accuracy": {str(k): v for k, v in self.per_class_accuracy.items()},
            "class_order": [str(k) for k in self.class_order],
            "n_candidates": len(self.labels),
        }
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        if plots:
            self._save_plots(out)

    def _save_plots(self, out: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(self.matrix, cmap="gnuplot", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax)
        ax.set_title("Normalized cross-correlation matrix")
        fig.savefig(out / "heatmap.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        iu = np.triu_indices_from(self.matrix, k=1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(np.asarray(self.matrix)[iu].ravel(), bins=100, color="steelblue")
        ax.set_xlabel("cross-correlation")
        ax.set_ylabel("frequency")
        fig.savefig(out / "score_hist.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


class Identifier:
    def __init__(
        self,
        library: CandidateLibrary,
        *,
        normalize: str = "symmetric",
        k: float = 2.0,
        backend: str = "auto",
        class_key_mode: str = "plane_burgers",
    ) -> None:
        self.library = library
        self.normalize = normalize
        self.k = k
        self.backend = backend
        self.class_key_mode = class_key_mode

    def rank(
        self, target_img: np.ndarray, target_pitch_um: tuple[float, float], *, top_k: int = 10
    ) -> list[RankedMatch]:
        t = resample_to_grid(target_img, target_pitch_um, self.library.grid)
        scores = score_target(
            t, self.library.frames, normalize=self.normalize, backend=self.backend, k=self.k
        )
        order = np.argsort(scores)[::-1][:top_k]
        return [
            RankedMatch(
                score=float(scores[i]),
                label=self.library.labels[i],
                scan_index=self.library.labels[i].scan_index,
                source_file=self.library.labels[i].source_file,
            )
            for i in order
        ]

    def study(self) -> IdentifiabilityResult:
        C = score_matrix(
            self.library.frames, normalize=self.normalize, backend=self.backend, k=self.k
        )
        keys = [lbl.class_key(self.class_key_mode) for lbl in self.library.labels]
        class_order = sorted(set(keys), key=str)
        idx_of = {key: i for i, key in enumerate(class_order)}
        n = len(keys)
        confusion = np.zeros((len(class_order), len(class_order)), dtype=int)
        per_tot = {key: 0 for key in class_order}
        per_ok = {key: 0 for key in class_order}
        correct = 0
        M = np.array(C, dtype=float, copy=True)
        np.fill_diagonal(M, -np.inf)  # leave-one-out: never match self
        for i in range(n):
            j = int(np.argmax(M[i]))
            true, pred = keys[i], keys[j]
            confusion[idx_of[true], idx_of[pred]] += 1
            per_tot[true] += 1
            if pred == true:
                correct += 1
                per_ok[true] += 1
        top1 = correct / n if n else 0.0
        per_class = {
            key: (per_ok[key] / per_tot[key] if per_tot[key] else 0.0) for key in class_order
        }
        return IdentifiabilityResult(
            matrix=C,
            labels=self.library.labels,
            top1_accuracy=top1,
            per_class_accuracy=per_class,
            confusion=confusion,
            class_order=class_order,
        )

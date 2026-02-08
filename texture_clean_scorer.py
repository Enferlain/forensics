import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

from typing import Optional


DEFAULT_METRICS = [
    "fg_flat_noise_tv",
    "fg_noise_tv",
    "fg_starved_ratio",
    "fg_edge_spread",
    "fg_mf_median",
]

DEFAULT_WEIGHTS = {
    "fg_flat_noise_tv": 1.0,
    "fg_noise_tv": 1.0,
    "fg_starved_ratio": 2.0,
    "fg_edge_spread": 0.5,
    "fg_mf_median": 0.5,
}


@dataclass
class RowScore:
    row: dict
    score_raw: float
    score_0_10: float


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    return mu, var**0.5


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    n = len(vs)
    mid = n // 2
    if n % 2 == 1:
        return float(vs[mid])
    return float((vs[mid - 1] + vs[mid]) / 2.0)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    if len(vs) == 1:
        return float(vs[0])
    rank = (p / 100.0) * (len(vs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(vs) - 1)
    frac = rank - lo
    return float(vs[lo] * (1.0 - frac) + vs[hi] * frac)


def robust_scale(values: list[float]) -> tuple[float, float]:
    med = _median(values)
    mad = _median([abs(v - med) for v in values])
    scale = 1.4826 * mad
    if scale <= 1e-8:
        _, std = _mean_std(values)
        scale = float(std)
    if scale <= 1e-8:
        scale = 1.0
    return med, scale


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def compute_scores_dataset(
    rows: list[dict],
    metrics: list[str],
    weights: Optional[dict[str, float]] = None,
) -> list[RowScore]:
    weights = weights or {}
    stats: dict[str, tuple[float, float]] = {}
    for m in metrics:
        vals = []
        for r in rows:
            raw = r.get(m, "")
            if raw == "":
                continue
            try:
                vals.append(float(raw))
            except ValueError:
                continue
        if not vals:
            raise ValueError(f"Metric '{m}' has no numeric values.")
        stats[m] = robust_scale(vals)

    raw_scores: list[float] = []
    scored: list[RowScore] = []
    for r in rows:
        vals = []
        for m in metrics:
            raw = r.get(m, "")
            if raw == "":
                vals = []
                break
            try:
                vals.append(float(raw))
            except ValueError:
                vals = []
                break
        if not vals:
            continue
        zsum = 0.0
        wsum = 0.0
        for v, m in zip(vals, metrics):
            med, scale = stats[m]
            w = float(weights.get(m, 1.0))
            zsum += ((v - med) / scale) * w
            wsum += w
        if wsum <= 1e-8:
            continue
        score_raw = -(zsum / wsum)
        raw_scores.append(score_raw)
        scored.append(RowScore(r, score_raw, 0.0))

    if not raw_scores:
        return []

    p5 = _percentile(raw_scores, 5.0)
    p95 = _percentile(raw_scores, 95.0)
    for rs in scored:
        if p95 > p5:
            norm = (rs.score_raw - p5) / (p95 - p5)
            if norm < 0.0:
                norm = 0.0
            elif norm > 1.0:
                norm = 1.0
            rs.score_0_10 = float(norm * 10.0)
        else:
            rs.score_0_10 = 5.0
    return scored


def score_lower_better(value: float, good: float, bad: float) -> float:
    if bad <= good:
        return 0.0
    t = (value - good) / (bad - good)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return float(1.0 - t)


def compute_scores_absolute(
    rows: list[dict],
    metrics: list[str],
    good_bad: dict[str, tuple[float, float]],
    weights: Optional[dict[str, float]] = None,
) -> list[RowScore]:
    weights = weights or {}
    scored: list[RowScore] = []
    for r in rows:
        vals = []
        for m in metrics:
            raw = r.get(m, "")
            if raw == "":
                vals = []
                break
            try:
                vals.append(float(raw))
            except ValueError:
                vals = []
                break
        if not vals:
            continue

        comps = []
        wsum = 0.0
        for v, m in zip(vals, metrics):
            good, bad = good_bad[m]
            w = float(weights.get(m, 1.0))
            comps.append(score_lower_better(v, good, bad) * w)
            wsum += w
        avg = (sum(comps) / wsum) if comps and wsum > 1e-8 else 0.0
        scored.append(RowScore(r, avg, avg * 10.0))
    return scored


def write_csv(
    rows: list[dict],
    scores: list[RowScore],
    out_path: Path,
) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    if "fg_texture_clean_score" not in fieldnames:
        fieldnames.append("fg_texture_clean_score")
    if "fg_texture_clean_score_0_10" not in fieldnames:
        fieldnames.append("fg_texture_clean_score_0_10")

    score_map = {id(rs.row): rs for rs in scores}
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            rs = score_map.get(id(r))
            if rs:
                r = dict(r)
                r["fg_texture_clean_score"] = f"{rs.score_raw:.6f}"
                r["fg_texture_clean_score_0_10"] = f"{rs.score_0_10:.6f}"
            writer.writerow(r)


def print_top(scores: list[RowScore], limit: int) -> None:
    if not scores:
        print("No scores computed.")
        return
    scores = sorted(scores, key=lambda s: s.score_raw, reverse=True)
    print("Top by fg_texture_clean_score:")
    for rs in scores[:limit]:
        image = rs.row.get("image", "")
        print(f"{image}\t{rs.score_raw:.4f}\t{rs.score_0_10:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute a texture-clean score from fg_* metrics."
    )
    parser.add_argument("--csv", type=str, required=True, help="Input CSV.")
    parser.add_argument("--out", type=str, default="", help="Output CSV path.")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Override metrics (default: fg_flat_noise_tv fg_noise_tv fg_starved_ratio fg_edge_spread fg_mf_median).",
    )
    parser.add_argument("--top", type=int, default=10, help="Print top N.")
    parser.add_argument(
        "--mode",
        choices=["absolute", "dataset"],
        default="absolute",
        help="Scoring mode: absolute (per-image) or dataset (normalized).",
    )
    parser.add_argument("--flat-noise-good", type=float, default=0.0045)
    parser.add_argument("--flat-noise-bad", type=float, default=0.0075)
    parser.add_argument("--noise-good", type=float, default=0.0045)
    parser.add_argument("--noise-bad", type=float, default=0.0075)
    parser.add_argument("--starved-good", type=float, default=0.020)
    parser.add_argument("--starved-bad", type=float, default=0.050)
    parser.add_argument("--edge-spread-good", type=float, default=0.100)
    parser.add_argument("--edge-spread-bad", type=float, default=0.130)
    parser.add_argument("--mf-median-good", type=float, default=0.018)
    parser.add_argument("--mf-median-bad", type=float, default=0.024)
    parser.add_argument("--w-flat-noise", type=float, default=DEFAULT_WEIGHTS["fg_flat_noise_tv"])
    parser.add_argument("--w-noise", type=float, default=DEFAULT_WEIGHTS["fg_noise_tv"])
    parser.add_argument("--w-starved", type=float, default=DEFAULT_WEIGHTS["fg_starved_ratio"])
    parser.add_argument("--w-edge-spread", type=float, default=DEFAULT_WEIGHTS["fg_edge_spread"])
    parser.add_argument("--w-mf-median", type=float, default=DEFAULT_WEIGHTS["fg_mf_median"])
    args = parser.parse_args()

    in_path = Path(args.csv)
    rows = load_rows(in_path)
    metrics = args.metrics if args.metrics else DEFAULT_METRICS
    weights = {
        "fg_flat_noise_tv": args.w_flat_noise,
        "fg_noise_tv": args.w_noise,
        "fg_starved_ratio": args.w_starved,
        "fg_edge_spread": args.w_edge_spread,
        "fg_mf_median": args.w_mf_median,
    }
    if args.mode == "dataset":
        scores = compute_scores_dataset(rows, metrics, weights)
    else:
        good_bad = {
            "fg_flat_noise_tv": (args.flat_noise_good, args.flat_noise_bad),
            "fg_noise_tv": (args.noise_good, args.noise_bad),
            "fg_starved_ratio": (args.starved_good, args.starved_bad),
            "fg_edge_spread": (args.edge_spread_good, args.edge_spread_bad),
            "fg_mf_median": (args.mf_median_good, args.mf_median_bad),
        }
        scores = compute_scores_absolute(rows, metrics, good_bad, weights)

    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_scored.csv")
    write_csv(rows, scores, out_path)
    print(f"Wrote: {out_path}")
    print_top(scores, args.top)


if __name__ == "__main__":
    main()

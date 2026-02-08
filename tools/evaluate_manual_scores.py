import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def mean(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def pearsonr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = 0.0
    dx = 0.0
    dy = 0.0
    for x, y in zip(xs, ys):
        ax = x - mx
        ay = y - my
        num += ax * ay
        dx += ax * ax
        dy += ay * ay
    den = math.sqrt(dx * dy)
    return (num / den) if den > 0 else 0.0


def rankdata(values: list[float]) -> list[float]:
    # Average ranks for ties (1..n)
    indexed = list(enumerate(values))
    indexed.sort(key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearmanr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    return pearsonr(rankdata(xs), rankdata(ys))


@dataclass(frozen=True)
class ThresholdReport:
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


def eval_threshold(
    manual_scores: list[float],
    model_scores: list[float],
    manual_good_ge: float,
    model_good_ge: float,
) -> ThresholdReport:
    tp = fp = tn = fn = 0
    for m, s in zip(manual_scores, model_scores):
        actual_good = m >= manual_good_ge
        pred_good = s >= model_good_ge
        if pred_good and actual_good:
            tp += 1
        elif pred_good and (not actual_good):
            fp += 1
        elif (not pred_good) and (not actual_good):
            tn += 1
        else:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return ThresholdReport(
        threshold=model_good_ge,
        precision=prec,
        recall=rec,
        f1=f1,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Join filename-based manual scores with scored metrics CSV and print correlation + threshold stats."
    )
    ap.add_argument("--manual", required=True, help="Manual scores CSV (image_name, image_path, manual_score).")
    ap.add_argument("--scored", required=True, help="Scored metrics CSV with an 'image' column.")
    ap.add_argument("--score-field", default="fg_texture_clean_score_0_10", help="Scorer field in the scored CSV.")
    ap.add_argument("--out", default="manual_scores_joined.csv", help="Output joined CSV.")
    ap.add_argument("--manual-good-ge", type=float, default=3.5, help="Manual score >= this is 'good'.")
    args = ap.parse_args()

    manual_path = Path(args.manual)
    scored_path = Path(args.scored)
    out_path = Path(args.out)

    manual_rows = load_csv(manual_path)
    scored_rows = load_csv(scored_path)

    manual_by_name: dict[str, dict] = {}
    for r in manual_rows:
        name = r.get("image_name", "")
        if name:
            manual_by_name[name] = r

    joined: list[dict] = []
    missing_manual = 0
    missing_score = 0
    for r in scored_rows:
        base = os.path.basename(r.get("image", ""))
        m = manual_by_name.get(base)
        if not m:
            missing_manual += 1
            continue
        try:
            ms = float(m.get("manual_score", "") or "")
        except ValueError:
            missing_score += 1
            continue
        try:
            ss = float(r.get(args.score_field, "") or "")
        except ValueError:
            missing_score += 1
            continue
        jr = {**r, **m, "basename": base, "manual_score_f": f"{ms:.6f}", "model_score_f": f"{ss:.6f}"}
        joined.append(jr)

    print(f"manual rows: {len(manual_rows)}")
    print(f"scored rows: {len(scored_rows)}")
    print(f"joined rows: {len(joined)}")
    print(f"scored rows missing manual score: {missing_manual}")
    print(f"rows with non-numeric score: {missing_score}")

    manual_scores = [float(r["manual_score_f"]) for r in joined]
    model_scores = [float(r["model_score_f"]) for r in joined]

    if joined:
        print("\nCorrelation:")
        print(f"pearson(manual, model) = {pearsonr(manual_scores, model_scores):.3f}")
        print(f"spearman(manual, model) = {spearmanr(manual_scores, model_scores):.3f}")

        # Simple “good” classification sweep over model score thresholds (0..10 step 0.1)
        best: ThresholdReport | None = None
        for i in range(0, 101):
            t = i / 10.0
            rep = eval_threshold(manual_scores, model_scores, args.manual_good_ge, t)
            if best is None or rep.f1 > best.f1:
                best = rep

        if best:
            print(f"\nGood thresholding (manual_good>={args.manual_good_ge}):")
            print(
                f"best model_good>={best.threshold:.1f} f1={best.f1:.3f} "
                f"prec={best.precision:.3f} rec={best.recall:.3f} "
                f"tp={best.tp} fp={best.fp} tn={best.tn} fn={best.fn}"
            )

        # Worst mismatches
        # High manual but low model
        hi_manual = sorted(joined, key=lambda r: (-float(r["manual_score_f"]), float(r["model_score_f"])))
        lo_manual = sorted(joined, key=lambda r: (float(r["manual_score_f"]), -float(r["model_score_f"])))
        print("\nTop 10: high manual, low model:")
        for r in hi_manual[:10]:
            print(f"{r['basename']}\tmanual={float(r['manual_score_f']):.3f}\tmodel={float(r['model_score_f']):.3f}")
        print("\nTop 10: low manual, high model:")
        for r in lo_manual[:10]:
            print(f"{r['basename']}\tmanual={float(r['manual_score_f']):.3f}\tmodel={float(r['model_score_f']):.3f}")

    write_csv(out_path, joined)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()

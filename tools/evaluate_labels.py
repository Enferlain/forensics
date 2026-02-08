import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ThresholdReport:
    threshold: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int


def percentile(values: list[float], p: float) -> float:
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


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def eval_clean_threshold(joined: list[dict], threshold: float, score_field: str) -> ThresholdReport:
    tp = fp = tn = fn = 0
    for r in joined:
        score = float(r.get(score_field, "0") or "0")
        pred_clean = score >= threshold
        actual_clean = r.get("label", "") == "clean"
        if pred_clean and actual_clean:
            tp += 1
        elif pred_clean and not actual_clean:
            fp += 1
        elif (not pred_clean) and (not actual_clean):
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return ThresholdReport(
        threshold=threshold,
        f1=f1,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Join label CSV with scored metrics CSV and print a small report.")
    ap.add_argument("--labels", required=True, help="Labels CSV (image_name,label,notes,subdir).")
    ap.add_argument("--scored", required=True, help="Scored metrics CSV (must include fg_texture_clean_score_0_10).")
    ap.add_argument("--out", default="labels_joined.csv", help="Output joined CSV.")
    ap.add_argument("--score-field", default="fg_texture_clean_score_0_10", help="Score field to evaluate.")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    scored_path = Path(args.scored)
    out_path = Path(args.out)

    labels_rows = load_csv(labels_path)
    scored_rows = load_csv(scored_path)

    labels_by_name = {r.get("image_name", ""): r for r in labels_rows if r.get("image_name", "")}

    joined: list[dict] = []
    for r in scored_rows:
        base = os.path.basename(r.get("image", ""))
        lab = labels_by_name.get(base)
        if not lab:
            continue
        joined.append({**r, **lab, "basename": base})

    scored_names = {os.path.basename(r.get("image", "")) for r in scored_rows}
    missing = [name for name in labels_by_name.keys() if name not in scored_names]

    print(f"labels rows: {len(labels_rows)}")
    print(f"scored rows: {len(scored_rows)}")
    print(f"joined rows: {len(joined)}")
    print(f"labels missing in scored: {len(missing)}")
    if missing:
        for n in missing[:10]:
            print(f"  missing: {n}")

    # Summaries by label
    by_label: dict[str, list[float]] = {}
    for r in joined:
        label = r.get("label", "")
        try:
            score = float(r.get(args.score_field, "0") or "0")
        except ValueError:
            continue
        by_label.setdefault(label, []).append(score)

    print(f"\nScore summary by label ({args.score_field}):")
    for label in sorted(by_label.keys()):
        scores = sorted(by_label[label])
        n = len(scores)
        mean = (sum(scores) / n) if n else 0.0
        print(
            f"{label:5} n={n:3} mean={mean:6.3f} "
            f"p10={percentile(scores, 10):6.3f} p50={percentile(scores, 50):6.3f} p90={percentile(scores, 90):6.3f}"
        )

    # Best threshold (0.0..10.0 step 0.1)
    best: ThresholdReport | None = None
    for i in range(0, 101):
        t = i / 10.0
        rep = eval_clean_threshold(joined, t, args.score_field)
        if best is None or rep.f1 > best.f1:
            best = rep

    if best:
        print("\nBest clean-vs-nonclean threshold by F1:")
        print(
            f"threshold={best.threshold:.1f} f1={best.f1:.3f} "
            f"prec={best.precision:.3f} rec={best.recall:.3f} acc={best.accuracy:.3f} "
            f"tp={best.tp} fp={best.fp} tn={best.tn} fn={best.fn}"
        )

    # Mismatches
    def score_of(row: dict) -> float:
        try:
            return float(row.get(args.score_field, "0") or "0")
        except ValueError:
            return 0.0

    clean = sorted([r for r in joined if r.get("label", "") == "clean"], key=score_of)
    nonclean = sorted([r for r in joined if r.get("label", "") != "clean"], key=score_of, reverse=True)
    print("\nLowest 10 clean scores:")
    for r in clean[:10]:
        print(f"{r.get('basename','')}\t{score_of(r):.3f}\t{r.get('notes','')}")
    print("\nHighest 10 nonclean scores:")
    for r in nonclean[:10]:
        print(f"{r.get('basename','')}\t{r.get('label','')}\t{score_of(r):.3f}\t{r.get('notes','')}")

    # Write joined csv
    for r in joined:
        # Ensure a stable numeric field for sorting/filtering in spreadsheets.
        r["label_score"] = f"{score_of(r):.6f}"
    write_csv(out_path, joined)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()

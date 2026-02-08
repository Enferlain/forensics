import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from blotch_blur_filter import BlotchBlurScorerV16


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class ScoreRow:
    path: Path
    texture: float
    detail: float
    line: float
    overall: float
    noise_factor: float | None = None
    flat_ratio: float | None = None
    starved_ratio: float | None = None


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTS:
            yield path
        return
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def score_image(scorer: BlotchBlurScorerV16, path: Path) -> ScoreRow:
    img = Image.open(path).convert("RGB")
    res = scorer.analyze(img)
    return ScoreRow(
        path=path,
        texture=res["texture_score"],
        detail=res["detail_score"],
        line=res["line_score"],
        overall=res["overall_score"],
        noise_factor=res.get("noise_factor"),
        flat_ratio=res.get("flat_ratio"),
        starved_ratio=res.get("starved_ratio"),
    )


def print_table(
    rows: list[ScoreRow],
    base_tex: float | None,
    base_det: float | None,
    debug: bool,
) -> None:
    if debug:
        header = (
            f"{'image':48}  {'tex':>6}  {'det':>6}  {'line':>6}  {'overall':>7}  "
            f"{'nfac':>6}  {'flat':>6}  {'starv':>6}  {'tex<':>5}  {'det<':>5}  {'both':>5}"
        )
    else:
        header = f"{'image':48}  {'tex':>6}  {'det':>6}  {'line':>6}  {'overall':>7}  {'tex<':>5}  {'det<':>5}  {'both':>5}"
    print(header)
    print("-" * len(header))
    for r in rows:
        tex_ok = (r.texture < base_tex) if base_tex is not None else False
        det_ok = (r.detail < base_det) if base_det is not None else False
        both_ok = tex_ok and det_ok
        if debug:
            nfac = f"{r.noise_factor:6.2f}" if r.noise_factor is not None else f"{'':>6}"
            flat = f"{r.flat_ratio:6.2f}" if r.flat_ratio is not None else f"{'':>6}"
            starv = f"{r.starved_ratio:6.2f}" if r.starved_ratio is not None else f"{'':>6}"
            print(
                f"{r.path.name[:48]:48}  "
                f"{r.texture:6.2f}  {r.detail:6.2f}  {r.line:6.2f}  {r.overall:7.2f}  "
                f"{nfac}  {flat}  {starv}  "
                f"{'Y' if tex_ok else 'N':>5}  {'Y' if det_ok else 'N':>5}  {'Y' if both_ok else 'N':>5}"
            )
        else:
            print(
                f"{r.path.name[:48]:48}  "
                f"{r.texture:6.2f}  {r.detail:6.2f}  {r.line:6.2f}  {r.overall:7.2f}  "
                f"{'Y' if tex_ok else 'N':>5}  {'Y' if det_ok else 'N':>5}  {'Y' if both_ok else 'N':>5}"
            )


def load_labels_csv(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        labels[parts[0]] = parts[1].lower()
    return labels


def auc_lower(bad: list[float], good: list[float]) -> float | None:
    total = len(bad) * len(good)
    if total == 0:
        return None
    wins = 0.0
    for b in bad:
        for g in good:
            if b < g:
                wins += 1.0
            elif b == g:
                wins += 0.5
    return wins / total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate subject texture/detail scores for a folder of images."
    )
    parser.add_argument("path", type=str, help="Image file or directory to scan.")
    parser.add_argument(
        "--baseline",
        nargs="+",
        default=[],
        help="Baseline image(s). Thresholds are min(texture/detail) across these.",
    )
    parser.add_argument(
        "--sort",
        choices=["overall", "detail", "texture", "line", "name"],
        default="overall",
        help="Sort output by a score column.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show extra columns used by the texture score.",
    )
    parser.add_argument(
        "--less-bad",
        nargs="*",
        default=[],
        help="Filenames for the 'less bad' group.",
    )
    parser.add_argument(
        "--worse",
        nargs="*",
        default=[],
        help="Filenames for the 'worse' group.",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help="CSV file with filename,label (label is 'less_bad' or 'worse').",
    )
    args = parser.parse_args()

    scorer = BlotchBlurScorerV16()

    base_tex = None
    base_det = None
    if args.baseline:
        base_rows = [score_image(scorer, Path(p)) for p in args.baseline]
        base_tex = min(r.texture for r in base_rows)
        base_det = min(r.detail for r in base_rows)
        print("Baseline scores:")
        for r in base_rows:
            if args.debug:
                print(
                    f"- {r.path.name}: tex={r.texture:.2f} det={r.detail:.2f} "
                    f"line={r.line:.2f} overall={r.overall:.2f} "
                    f"nfac={r.noise_factor:.2f} flat={r.flat_ratio:.2f} starv={r.starved_ratio:.2f}"
                )
            else:
                print(
                    f"- {r.path.name}: tex={r.texture:.2f} det={r.detail:.2f} line={r.line:.2f} overall={r.overall:.2f}"
                )
        print(f"Thresholds: tex < {base_tex:.2f}, det < {base_det:.2f}\n")

    rows = [score_image(scorer, p) for p in iter_images(Path(args.path))]

    if args.sort == "name":
        rows.sort(key=lambda r: r.path.name.lower())
    elif args.sort == "texture":
        rows.sort(key=lambda r: r.texture)
    elif args.sort == "detail":
        rows.sort(key=lambda r: r.detail)
    elif args.sort == "line":
        rows.sort(key=lambda r: r.line)
    else:
        rows.sort(key=lambda r: r.overall)

    print_table(rows, base_tex, base_det, args.debug)

    if base_tex is not None and base_det is not None:
        both = [r for r in rows if r.texture < base_tex and r.detail < base_det]
        print(f"\nPass (both lower): {len(both)}/{len(rows)}")

    labels: dict[str, str] = {}
    if args.labels_csv:
        labels.update(load_labels_csv(Path(args.labels_csv)))
    for name in args.less_bad:
        labels[name] = "less_bad"
    for name in args.worse:
        labels[name] = "worse"

    if labels:
        by_label: dict[str, list[ScoreRow]] = {"less_bad": [], "worse": []}
        for r in rows:
            label = labels.get(r.path.name)
            if label in by_label:
                by_label[label].append(r)

        less_bad = by_label["less_bad"]
        worse = by_label["worse"]
        if less_bad and worse:
            def mean(vals: list[float]) -> float:
                return sum(vals) / len(vals)

            def collect(rows_in: list[ScoreRow], attr: str) -> list[float]:
                return [getattr(r, attr) for r in rows_in]

            print("\nGroup stats:")
            for label, group in [("less_bad", less_bad), ("worse", worse)]:
                print(
                    f"- {label}: n={len(group)} "
                    f"tex={mean(collect(group, 'texture')):.2f} "
                    f"det={mean(collect(group, 'detail')):.2f} "
                    f"line={mean(collect(group, 'line')):.2f} "
                    f"overall={mean(collect(group, 'overall')):.2f}"
                )

            print("\nSeparation (AUC where lower=more bad):")
            for metric in ["texture", "detail", "line", "overall"]:
                auc = auc_lower(
                    collect(worse, metric),
                    collect(less_bad, metric),
                )
                if auc is not None:
                    print(f"- {metric}: {auc:.3f}")
        else:
            print("\nLabels provided, but one of the groups is empty. Check filenames.")


if __name__ == "__main__":
    main()

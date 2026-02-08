import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CSV_CANDIDATES = [
    "image_quality_metrics_2.csv",
    "image_quality_metrics.csv",
]


def pick_default_csv() -> Path | None:
    for name in DEFAULT_CSV_CANDIDATES:
        path = Path(name)
        if path.exists():
            return path
    return None


def safe_name(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def load_metrics(csv_path: Path) -> Tuple[List[str], Dict[str, np.ndarray]]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV is empty.")

    fieldnames = reader.fieldnames or []
    images: List[str] = []
    for r in rows:
        raw = r.get("image", "")
        images.append(Path(raw).name if raw else f"row_{len(images)}")

    numeric: Dict[str, np.ndarray] = {}
    for col in fieldnames:
        if col == "image":
            continue
        values: List[float] = []
        any_value = False
        for r in rows:
            s = r.get(col, "")
            if s is None or s == "":
                values.append(np.nan)
                continue
            try:
                values.append(float(s))
                any_value = True
            except ValueError:
                values.append(np.nan)
        if any_value:
            numeric[col] = np.array(values, dtype=float)

    return images, numeric


def resolve_image_path(raw: str) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p
    raw_norm = raw.replace("\\", "/")
    if len(raw_norm) >= 2 and raw_norm[1] == ":":
        drive = raw_norm[0].lower()
        rest = raw_norm[2:].lstrip("/")
        mapped = Path(f"/mnt/{drive}/{rest}")
        if mapped.exists():
            return mapped
    return None


def render_metric_grid(
    metric: str,
    items: List[Tuple[str, float]],
    out_dir: Path,
    thumb: int,
    columns: int,
    max_images: int,
    descending: bool,
    font_size: int,
) -> None:
    values = [(name, val) for name, val in items if np.isfinite(val)]
    if not values:
        return
    values.sort(key=lambda x: x[1], reverse=descending)
    if max_images > 0:
        values = values[:max_images]

    rows = math.ceil(len(values) / columns)
    cell_w = thumb
    cell_h = thumb + font_size + 6
    grid = Image.new("RGB", (columns * cell_w, rows * cell_h), (20, 20, 20))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for idx, (name, val) in enumerate(values):
        row = idx // columns
        col = idx % columns
        x0 = col * cell_w
        y0 = row * cell_h

        img_path = resolve_image_path(name)
        if img_path and img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((thumb, thumb))
                pad_x = x0 + (thumb - img.width) // 2
                pad_y = y0 + (thumb - img.height) // 2
                grid.paste(img, (pad_x, pad_y))
            except Exception:
                draw.rectangle(
                    [x0 + 2, y0 + 2, x0 + thumb - 2, y0 + thumb - 2],
                    outline=(200, 60, 60),
                    width=2,
                )
        else:
            draw.rectangle(
                [x0 + 2, y0 + 2, x0 + thumb - 2, y0 + thumb - 2],
                outline=(200, 60, 60),
                width=2,
            )

        label = f"{Path(name).name} | {val:.4f}"
        draw.text((x0 + 4, y0 + thumb + 2), label, fill=(230, 230, 230), font=font)

    out_path = out_dir / f"grid_{safe_name(metric)}.png"
    grid.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize numeric metrics from the image quality CSV."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV (default: image_quality_metrics_2.csv or image_quality_metrics.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="metric_plots",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=5,
        help="Number of columns in image grids.",
    )
    parser.add_argument(
        "--thumb",
        type=int,
        default=256,
        help="Thumbnail size in pixels.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Max images per grid (0 = all).",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort grids in descending order of the metric.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Limit to specific metrics (by column name).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else pick_default_csv()
    if csv_path is None or not csv_path.exists():
        raise SystemExit("CSV not found. Use --csv to specify the file.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images, metrics = load_metrics(csv_path)
    allowed = set(args.metrics) if args.metrics else None

    for name, values in metrics.items():
        if allowed is not None and name not in allowed:
            continue
        items = list(zip(images, values.tolist()))
        render_metric_grid(
            name,
            items,
            out_dir,
            thumb=args.thumb,
            columns=args.columns,
            max_images=args.max_images,
            descending=args.descending,
            font_size=12,
        )

    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()

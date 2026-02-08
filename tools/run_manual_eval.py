import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("\n$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end eval: compute metrics -> score -> compare vs manual filename scores."
    )
    ap.add_argument(
        "--images",
        default=r"D:\stable-diffusion-webui-reForge\extensions\sd-optim\logs\2026-02-04_15-22-54_pop_lora_['manual']\imgs",
        help="Folder (or file) containing images to evaluate.",
    )
    ap.add_argument(
        "--manual",
        default=str(Path("labels") / "manual_scores_2026-02-04_15-22-54.csv"),
        help="CSV with (image_name,image_path,manual_score).",
    )
    ap.add_argument(
        "--out-prefix",
        default=str(Path("outputs") / "image_quality_metrics_manual"),
        help="Prefix for generated output files (no extension).",
    )
    ap.add_argument(
        "--manual-good-ge",
        type=float,
        default=3.5,
        help="Manual score >= this is considered 'good' when sweeping model thresholds.",
    )
    ap.add_argument(
        "--u2net-home",
        default="",
        help="Override U2NET_HOME (rembg model cache directory). Recommended to set to a writable folder.",
    )
    ap.add_argument(
        "--rembg-model",
        default="u2net",
        help="rembg model name (e.g. u2net, isnet-anime, bria-rmbg).",
    )
    ap.add_argument(
        "--no-rembg-post-process",
        action="store_true",
        help="Disable rembg post processing of masks.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    args = ap.parse_args()

    # Repo root (this file lives in tools/).
    repo = Path(__file__).resolve().parents[1]
    out_prefix = Path(args.out_prefix)
    metrics_csv = (repo / out_prefix).with_suffix(".csv")
    scored_csv = (repo / out_prefix.parent / f"{out_prefix.name}_scored.csv")
    joined_csv = repo / Path("outputs") / "manual_scores_joined.csv"

    env = dict(os.environ)
    if args.u2net_home:
        env["U2NET_HOME"] = args.u2net_home

    # 1) Metrics (with rembg fg_* columns)
    cmd_metrics = [
        sys.executable,
        str(repo / "image_quality_metrics.py"),
        "--paths",
        args.images,
        "--use-rembg",
        "--rembg-model",
        args.rembg_model,
        "--out",
        str(metrics_csv),
        "--limit",
        "0",
    ]
    if args.no_rembg_post_process:
        cmd_metrics.append("--no-rembg-post-process")

    # 2) Score
    cmd_score = [
        sys.executable,
        str(repo / "texture_clean_scorer.py"),
        "--csv",
        str(metrics_csv),
        "--out",
        str(scored_csv),
        "--top",
        "10",
    ]

    # 3) Evaluate vs manual
    cmd_eval = [
        sys.executable,
        str(repo / "evaluate_manual_scores.py"),
        "--manual",
        str((repo / args.manual) if not os.path.isabs(args.manual) else args.manual),
        "--scored",
        str(scored_csv),
        "--out",
        str(joined_csv),
        "--manual-good-ge",
        str(args.manual_good_ge),
    ]

    print(f"Outputs:\n- {metrics_csv}\n- {scored_csv}\n- {joined_csv}")
    run(cmd_metrics, env, args.dry_run)
    run(cmd_score, env, args.dry_run)
    run(cmd_eval, env, args.dry_run)


if __name__ == "__main__":
    main()

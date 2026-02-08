Forensic filters and scoring for image analysis.

**Repo Layout**
- `image_quality_metrics.py`: Compute global metrics plus optional foreground (`fg_*`) metrics; outputs CSV.
- `texture_clean_scorer.py`: Convert selected `fg_*` metrics into `fg_texture_clean_score_0_10`.
- `tools/`: Utilities for label joins, manual-score eval, and metric visualizations.
- `labels/`: Human labels and manual scores for evaluation.
- `outputs/`: Generated CSVs from metrics, scoring, and evaluation (scratch; gitignored).
- `test/`, `test_new/`: Image folders used by scripts.

**Setup Notes**
Examples use `uv run python ...`. If `rembg` complains about model cache permissions, set `U2NET_HOME` or pass `--u2net-home` to a writable folder containing the `u2net.onnx` model.

**Typical Workflow**
1. Compute metrics (optionally with foreground segmentation):
```powershell
uv run python image_quality_metrics.py --paths "D:\Projects\forensics\test_new" --use-rembg --out outputs/image_quality_metrics_test_new.csv
```
2. Score the CSV with texture-clean scoring:
```powershell
uv run python texture_clean_scorer.py --csv outputs/image_quality_metrics_test_new.csv --out outputs/image_quality_metrics_test_new_scored.csv
```
3. Evaluate against human labels:
```powershell
uv run python evaluate_labels.py --labels labels/labels_test_new_2048.csv --scored outputs/image_quality_metrics_test_new_scored.csv --out outputs/test_new_labels_joined.csv
```
4. End-to-end manual-score evaluation (144 images):
```powershell
uv run python run_manual_eval.py --u2net-home "D:\Projects\forensics\.u2net"
```

**Scripts**
**`image_quality_metrics.py`**
Computes no-reference image quality metrics for all images in `--paths` or, by default, image files in the repo root plus `test/` and `tests/`. With `--use-rembg`, it also computes foreground-only (`fg_*`) metrics and derives a `fg_texture_clean_score_0_10`.
Typical run:
```powershell
uv run python image_quality_metrics.py --paths "D:\Projects\forensics\test_new" --use-rembg --out outputs/image_quality_metrics_test_new.csv
```

**`test_bg.py`**
Analyzes background blackness by removing the subject with `rembg`, finding the dominant background color (via KMeans), and scoring proximity to black. Visualizes the original image, background-removed image, and dominant color swatch.
Typical run:
```powershell
uv run python test_bg.py "D:\Projects\forensics\test_new\some_image.png"
```

**`test_noise.py`**
Two-stage background noise scorer. Uses `rembg` to sample background, clusters for the dominant background color, builds a precise color-based background mask, and scores noise in the background. Visualizes original, isolated background, and noise map.
Typical run:
```powershell
uv run python test_noise.py "D:\Projects\forensics\test_new\some_image.png"
```

**`test_pca.py`**
PCA-based noise scorer. Runs PCA on pixel data and scores noise from the PCA projection. If no image path is supplied, it generates a synthetic test image. Visualizes original and PCA result.
Typical run with an image:
```powershell
uv run python test_pca.py "D:\Projects\forensics\test_new\some_image.png"
```
Typical run without an image:
```powershell
uv run python test_pca.py
```

**`test_texture.py`**
Standalone, single-image texture-clean scorer with visualization. Uses `rembg` for subject mask and computes `fg_*` texture metrics (noise, starved ratio, edge spread, mid-frequency energy). Prints a 0..10 score plus per-metric parts and shows diagnostic maps unless `--no-plot` is used.
Typical run:
```powershell
uv run python test_texture.py "D:\Projects\forensics\test_new\some_image.png"
```
Disable plotting:
```powershell
uv run python test_texture.py "D:\Projects\forensics\test_new\some_image.png" --no-plot
```

**`test_texture copy.py`**
Legacy/compatibility version of the texture-clean visualizer. It imports scoring defaults from `image_quality_metrics.py`, which are not currently defined there, so it may need import updates to run. Name contains a space, so quote the path.
Typical run:
```powershell
uv run python "test_texture copy.py" "D:\Projects\forensics\test_new\some_image.png"
```

**`test_texture_old.py`**
Older high-sensitivity scorer. Uses TV denoising, flat-zone noise estimates, and a sigmoid sharpness term to produce a score; visualizes subject mask, flat zones, and residual noise.
Typical run:
```powershell
uv run python test_texture_old.py "D:\Projects\forensics\test_new\some_image.png"
```

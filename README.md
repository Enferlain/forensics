This is for making forensic filters for analysis and scoring of images.

**Repo Layout**
- `image_quality_metrics.py`: Computes global + `fg_*` (subject) metrics to CSV.
- `texture_clean_scorer.py`: Turns selected `fg_*` metrics into `fg_texture_clean_score_0_10`.
- `tools/`: Utilities (label joins, manual-score eval, metric visualizers).
- `labels/`: Human labels / manual scores used for evaluation.
- `outputs/`: Generated CSVs from metrics/scoring/evaluation (scratch output; gitignored).
- `test/`, `test_new/`: Image folders.

**Common Commands (Windows / PowerShell)**
- Compute metrics:
- `uv run python image_quality_metrics.py --paths "D:\Projects\forensics\test_new" --use-rembg --out outputs/image_quality_metrics_test_new.csv`
- Score the CSV:
- `uv run python texture_clean_scorer.py --csv outputs/image_quality_metrics_test_new.csv --out outputs/image_quality_metrics_test_new_scored.csv`
- Evaluate against human labels:
- `uv run python evaluate_labels.py --labels labels/labels_test_new_2048.csv --scored outputs/image_quality_metrics_test_new_scored.csv --out outputs/test_new_labels_joined.csv`
- End-to-end manual-score evaluation (144 images):
- `uv run python run_manual_eval.py --u2net-home "D:\Projects\forensics\.u2net"`

If `rembg` complains about model cache permissions, set `U2NET_HOME` (or pass `--u2net-home`) to a writable folder that already contains `u2net.onnx` / related models.

---

original test images in test folder:

all images in the test folder are noisy, blurry, and have noisy/blurry/smudged details and texture, sometimes on both the subject and the background, and sometimes mostly the background. 

noisy but less blurry and smudgy 000-00-shiro_black-4.966.png 010-00-shiro_black-5.057.png 011-00-shiro_black-5.882.png 116-00-shiro_black-7.595.png 159-00-shiro_black-6.230.png 199-00-shiro_black-5.248.png 203-00-shiro_black-6.077.png 245-00-shiro_black-5.122.png 310-00-shiro_black-6.733.png

noisy and very smudgy and blurry - 242-00-shiro_black-5.456.png 271-00-shiro_black-5.009.png 294-00-shiro_black-4.710.png

038-04-noob10b-6.081.png is 1536x2048 and noisy overall (invisible mostly), but not blurry on details, and there are no visible texture artifacts

00166-4154668875.png is clean, should score the highest for texture and subject quality (other than 0.38 but that one is 2x pixels)

---

rule for new test images (manual labels) in test_new

clean:

Subject edges look crisp at 100% zoom.
No visible smudging/texture mush in key areas (face/hair/clothing edges).
Noise/grain is minimal or only in background.
You would not hesitate to call it “good quality.”

ok:

Subject is mostly clear, but one noticeable issue: light blur, mild smudge, or mild noise on subject.
Edges are soft but still acceptable.
Would pass as “usable,” not “clean.”

bad:

Subject has obvious blur/smudge/noise.
Fine texture is lost (hair strands, fabric detail), or edges look washed out.
You would call it “poor quality.”

"""Stage 6, step 1: organize the plant photos for COLMAP and training.

    python -m scripts.prepare_plant --src data/plant/pilot --dst data/plant

For every JPEG in ``--src`` (other files are ignored), in capture order (file names encode
the capture time):
  * apply the EXIF orientation to the pixels (COLMAP ignores the EXIF flag) and drop it,
  * write   images_full/frame_XXX.jpg  (original resolution, quality 95)
            images_2000/frame_XXX.jpg  (long side 2000 px)
            images_1600/frame_XXX.jpg  (long side 1600 px)
    with area (box) filtering. Downscaling by s maps continuous pixel coordinates exactly
    u' = s u (origin at the image corner), so intrinsics scale as K' = diag(s, s, 1) K.
  * record frames.csv: new name, original name, EXIF time, sharpness.

Sharpness = variance of the Laplacian of the grey 1600-px image (higher = sharper); frames
far below the median are flagged as possibly blurred.
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SIZES = {"images_2000": 2000, "images_1600": 1600}


def process(job):
    i, src, dst = job
    im = ImageOps.exif_transpose(Image.open(src))
    exif_time = Image.open(src).getexif().get_ifd(0x8769).get(36867, "")
    rgb = np.asarray(im.convert("RGB"))
    name = f"frame_{i:03d}.jpg"
    Image.fromarray(rgb).save(dst / "images_full" / name, quality=95, subsampling=0)
    sharp = None
    for folder, long_side in SIZES.items():
        h, w = rgb.shape[:2]
        s = long_side / max(h, w)
        small = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        Image.fromarray(small).save(dst / folder / name, quality=95, subsampling=0)
        if folder == "images_1600":
            grey = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            sharp = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    return name, src.name, exif_time, sharp, rgb.shape[1], rgb.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    for folder in ["images_full", *SIZES]:
        (dst / folder).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=14) as ex:
        rows = list(ex.map(process, [(i, f, dst) for i, f in enumerate(files)]))
    with open(dst / "frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "original", "exif_time", "sharpness", "width", "height"])
        w.writerows(rows)
    sharp = np.array([r[3] for r in rows])
    med = np.median(sharp)
    print(f"{len(rows)} frames, {rows[0][4]}x{rows[0][5]} -> {', '.join(SIZES)}")
    print(f"sharpness: median {med:.0f}, min {sharp.min():.0f}, max {sharp.max():.0f}")
    for r in rows:
        if r[3] < 0.4 * med:
            print(f"  possibly blurred: {r[0]} ({r[1]}), sharpness {r[3]:.0f}")


if __name__ == "__main__":
    main()

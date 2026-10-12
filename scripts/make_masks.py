"""Stage 6: foreground masks (plant + pot + saucer) with BiRefNet via rembg.

    U2NET_HOME=/mnt/cache/u2net python -m scripts.make_masks --scene data/plant

BiRefNet resizes its input to 1024x1024 internally, so it is run once per frame on the
2000-px image and the soft mask (uint8 alpha, 255 = foreground) is resized to the other
resolutions with bilinear filtering:
    masks_2000/frame_XXX.png, masks_1600/..., masks_full/...
Runs on the CPU: this onnxruntime build needs CUDA 13 libraries the env doesn't have.
Already existing masks are skipped, so the script can be re-run after an interruption.
"""
import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

FOLDERS = {"masks_2000": "images_2000", "masks_1600": "images_1600", "masks_full": "images_full"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/plant")
    ap.add_argument("--model", default="birefnet-general")
    args = ap.parse_args()
    scene = Path(args.scene)
    for m in FOLDERS:
        (scene / m).mkdir(exist_ok=True)
    from rembg import new_session, remove
    sess = new_session(args.model, providers=["CPUExecutionProvider"])
    frames = sorted((scene / "images_2000").glob("*.jpg"))
    t0 = time.time()
    for k, f in enumerate(frames):
        if all((scene / m / f.with_suffix(".png").name).exists() for m in FOLDERS):
            continue
        mask = remove(Image.open(f).convert("RGB"), session=sess, only_mask=True)   # 'L', same size
        for m, img_folder in FOLDERS.items():
            size = Image.open(scene / img_folder / f.name).size
            out = mask if mask.size == size else mask.resize(size, Image.BILINEAR)
            out.save(scene / m / f.with_suffix(".png").name)
        fg = (np.asarray(mask) > 127).mean()
        print(f"[{k + 1:3d}/{len(frames)}] {f.name}  foreground {fg:.3f}  {time.time() - t0:6.0f}s", flush=True)


if __name__ == "__main__":
    main()

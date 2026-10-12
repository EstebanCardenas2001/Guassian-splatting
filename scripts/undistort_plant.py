"""Stage 6: undistort images and masks to an ideal pinhole camera, at every resolution.

    python -m scripts.undistort_plant --scene data/plant --model colmap_incremental/sparse/1

COLMAP's OPENCV model maps a pinhole point (x, y) = (X/Z, Y/Z) to distorted
    r^2 = x^2 + y^2,  g = 1 + k1 r^2 + k2 r^4
    x_d = g x + 2 p1 x y + p2 (r^2 + 2 x^2),   y_d = g y + p1 (r^2 + 2 y^2) + 2 p2 x y
    u = fx x_d + cx,  v = fy y_d + cy.
My renderer is a pure pinhole, so every image is resampled: for each output pixel of the new
pinhole camera K_new, find the distorted source location and interpolate (cv2.remap,
bilinear). K_new comes from cv2.getOptimalNewCameraMatrix(alpha=0): the largest pinhole
view that contains only valid source pixels (no black corners), at the original size.

The model was estimated on the 2000-px images. Images at scale s relative to it have
K = diag(s, s, 1) K_2000 exactly (pixel origin at the corner), and the distortion
coefficients are dimensionless, so one model serves all resolutions.
Writes <scene>/undistorted/{images,masks}_{1600,2000,full}/ and cameras.json.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pycolmap

RES = {"2000": "images_2000", "1600": "images_1600", "full": "images_full"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/plant")
    ap.add_argument("--model", default="colmap_incremental/sparse/1")
    args = ap.parse_args()
    scene = Path(args.scene)
    rec = pycolmap.Reconstruction(str(scene / args.model))
    (cam,) = rec.cameras.values()
    assert cam.model.name == "OPENCV", cam.model
    fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
    W0, H0 = cam.width, cam.height
    dist = np.array([k1, k2, p1, p2])
    out_root = scene / "undistorted"
    meta = {"source_model": args.model, "distortion_k1k2p1p2": dist.tolist(), "resolutions": {}}

    for key, img_dir in RES.items():
        names = sorted((scene / img_dir).glob("*.jpg"))
        W, H = cv2.imread(str(names[0])).shape[1::-1]
        s = W / W0
        assert abs(H / H0 - s) < 1e-3, (W, H, W0, H0)
        K = np.array([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1]])
        K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), alpha=0, newImgSize=(W, H))
        mx, my = cv2.initUndistortRectifyMap(K, dist, None, K_new, (W, H), cv2.CV_32FC1)
        (out_root / f"images_{key}").mkdir(parents=True, exist_ok=True)
        (out_root / f"masks_{key}").mkdir(parents=True, exist_ok=True)

        def work(p):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            cv2.imwrite(str(out_root / f"images_{key}" / p.name),
                        cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            m = cv2.imread(str(scene / f"masks_{key}" / p.with_suffix(".png").name), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(out_root / f"masks_{key}" / p.with_suffix(".png").name),
                        cv2.remap(m, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0))

        with ThreadPoolExecutor(14) as ex:
            list(ex.map(work, names))
        meta["resolutions"][key] = {"width": W, "height": H, "scale_vs_model": s,
                                    "K": K_new.tolist(), "K_distorted": K.tolist()}
        print(f"{key:5s} {W}x{H}  s={s:.4f}  f {K[0, 0]:.1f} -> {K_new[0, 0]:.1f} px, "
              f"c ({K[0, 2]:.1f},{K[1, 2]:.1f}) -> ({K_new[0, 2]:.1f},{K_new[1, 2]:.1f})  {len(names)} frames")
    (out_root / "cameras.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

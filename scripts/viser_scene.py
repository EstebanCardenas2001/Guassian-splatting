"""Interactive 3D check of the plant reconstruction in the browser (viser).

    python -m scripts.viser_scene [--port 8080]
Then open http://localhost:8080 (VS Code forwards the port; otherwise
`ssh -L 8080:localhost:8080 <vm>`). Shows the normalized SfM point cloud (plant at the
origin, +y up) and every camera as a frustum textured with its photo: train cameras blue,
held-out test cameras orange. Click a frustum to jump to that viewpoint.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import viser
import viser.transforms as vtf

from src.camera import load_colmap

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max_points", type=int, default=150_000)
    args = ap.parse_args()
    d = load_colmap(ROOT / "data/plant", res="1600", load_images=True)
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+y")

    P, C = d["points"], d["point_colors"]
    keep = np.linalg.norm(P, axis=1) < 6.0
    server.scene.add_point_cloud("/sfm_points", points=P[keep][: args.max_points],
                                 colors=C[keep][: args.max_points], point_size=0.01)

    for split, color in (("train", (42, 120, 214)), ("test", (235, 104, 52))):
        for cam, img in zip(d[f"{split}_cams"], d[f"{split}_images"]):
            R, t = cam.viewmat[:3, :3].numpy(), cam.viewmat[:3, 3].numpy()
            c2w_R, c2w_t = R.T, -R.T @ t                           # camera -> world
            fy = cam.K[1, 1].item()
            thumb = img.numpy()[::8, ::8]
            frustum = server.scene.add_camera_frustum(
                f"/{split}/{cam.name}", fov=2 * np.arctan2(cam.height / 2, fy),
                aspect=cam.width / cam.height, scale=0.25, image=thumb, color=color,
                wxyz=vtf.SO3.from_matrix(c2w_R).wxyz, position=c2w_t)

            @frustum.on_click
            def _(event, wxyz=vtf.SO3.from_matrix(c2w_R).wxyz, pos=c2w_t):
                for client in server.get_clients().values():
                    client.camera.wxyz, client.camera.position = wxyz, pos
    print(f"viser on http://localhost:{args.port}  ({keep.sum()} points, "
          f"{len(d['train_cams'])} train + {len(d['test_cams'])} test cameras)", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()

"""Interactive viewer for a trained model: gsplat renders on the server, viser streams it.

    python -m scripts.view_splats --run outputs/<run> [--port 8080]
Open http://localhost:8080 (VS Code forwards the port). Drag to orbit, scroll to zoom.
The full model is rendered (incl. spherical harmonics, so the leaf sheen changes with the
viewpoint), not a browser approximation.

viser reports each client's camera as camera-to-world (wxyz, position) in the OpenCV
convention (x right, y down, z forward) plus a vertical field of view, so
viewmat = [R^T | -R^T p] and fy = (H/2) / tan(fov/2).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import viser
import viser.transforms as vtf

from src.gaussians import Gaussians
from src.render_gsplat import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    ck = torch.load(Path(args.run) / "ckpt.pt", map_location="cuda", weights_only=False)
    model = Gaussians.from_state_dict(ck["model"]).cuda()
    geo = [t.detach() for t in model.geometry()]
    print(f"{len(model)} Gaussians, SH degree {model.sh_degree}", flush=True)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+y")
    res = server.gui.add_slider("Max resolution", min=320, max=1600, step=80, initial_value=960)
    bg_choice = server.gui.add_dropdown("Background", ("white", "black", "grey"), initial_value="white")
    info = server.gui.add_markdown("")
    BG = {"white": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0), "grey": (0.5, 0.5, 0.5)}

    @torch.no_grad()
    def draw(client: viser.ClientHandle):
        cam = client.camera
        aspect = cam.aspect
        H = int(res.value) if aspect < 1 else int(res.value / aspect)
        W = int(round(H * aspect))
        R = vtf.SO3(np.asarray(cam.wxyz)).as_matrix()            # camera -> world (OpenCV axes)
        p = np.asarray(cam.position)
        V = torch.eye(4, device="cuda")
        V[:3, :3] = torch.as_tensor(R.T, dtype=torch.float32)
        V[:3, 3] = torch.as_tensor(-R.T @ p, dtype=torch.float32)
        f = 0.5 * H / np.tan(0.5 * cam.fov)
        K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device="cuda", dtype=torch.float32)
        t0 = time.perf_counter()
        img, _, _ = render(*geo, model.colors_for(V), V, K, W, H,
                           bg=torch.tensor(BG[bg_choice.value], device="cuda"))
        torch.cuda.synchronize()
        info.content = f"{len(model):,} Gaussians · {W}×{H} · {1e3 * (time.perf_counter() - t0):.0f} ms"
        client.scene.set_background_image((img.clamp(0, 1) * 255).byte().cpu().numpy(), format="jpeg",
                                          jpeg_quality=90)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = (0.0, 1.2, -3.2)                # start: in front, slightly above
        client.camera.look_at = (0.0, 0.0, 0.0)
        client.camera.up_direction = (0.0, 1.0, 0.0)

        @client.camera.on_update
        def _(_):
            draw(client)
        draw(client)

    for h in (res, bg_choice):
        h.on_update(lambda _: [draw(c) for c in server.get_clients().values()])
    print(f"viewer on http://localhost:{args.port}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()

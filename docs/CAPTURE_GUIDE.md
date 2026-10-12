# Photo capture guide: the pothos for Gaussian Splatting

How to photograph the plant so that COLMAP can find the camera poses and the Gaussians can
reconstruct sharp leaves, thin stems and the terracotta texture.

**In short:** ~170–190 photos, taken while you **walk around a perfectly still plant** in
5 rings at different heights. Keep the **same distance, focus, zoom, exposure and white
balance for every photo**, use soft constant light, and put the camera on a tripod.

---

## 1. Checklist (print this or keep it on your phone)

**Before you start**
- [ ] Plant on the floor or a low stool, in a room with **no airflow** (fans, AC, open windows off/closed)
- [ ] Soft, constant light: overcast window light or 2 diffused lamps; **no flash, no direct sun**
- [ ] Plain matte background that is **not green and not orange** (light grey or white sheet)
- [ ] A few textured flat objects (newspaper, magazine) on the floor around the pot, *not* touching it
- [ ] Tape marks on the floor for each ring (Section 3)

**Camera (Nikon D7200)**
- [ ] Mode **M**, **f/8** (f/11 if the plant is deeper than ~50 cm), **ISO 100–200**, shutter whatever exposes correctly
- [ ] **Tripod** + 2 s self-timer or remote; **Exposure delay mode ON** (mirror slap)
- [ ] **VR (stabilization) OFF** on the lens
- [ ] Focal length **35 mm, taped** if it's a zoom; never touch the zoom ring again
- [ ] Autofocus once on the middle of the plant, then switch the lens/body to **MF**; never refocus
- [ ] White balance **fixed** (preset from a grey card, or a fixed Kelvin value), never Auto
- [ ] Picture Control **Neutral**; **Active D-Lighting OFF**; **Auto distortion control OFF**; vignette control OFF
- [ ] Image quality **RAW + JPEG Fine, Large**
- [ ] Highlight warning (blinkies) ON; check that no leaf highlight or pot rim is clipped

**Shooting**
- [ ] Pilot: 1 ring of ~20 photos → upload → I check them (Section 6) → then the full capture
- [ ] 5 rings at a constant camera-to-plant distance, every photo framing the **whole plant and pot**
- [ ] Never move or rotate the plant, never change settings, keep your shadow off the plant

---

## 2. How many photos

| ring | elevation | camera position | photos | step between photos |
|---|---|---|---|---|
| 1 low | ~0° (level with the middle of the pot) | sees the pot sides and the underside of hanging vines | 40 | 9° |
| 2 | ~20° | | 40 | 9° |
| 3 | ~40° | | 40 | 9° |
| 4 | ~60° | | 32 | 11° |
| 5 top | ~80° (almost straight down) | | 12 | 30° |
| extra | any ring | gaps around dense foliage or long trailing vines, same distance | 0–30 | as needed |
| **total** | | | **~165–195** | |

**Why these numbers**
- **Overlap.** Orbiting keeps the plant centred in every frame. With a 9° step, a leaf
  facing outwards appears in about 10 consecutive photos of a ring (all those within ±45°
  of facing it). COLMAP's SIFT matching stays reliable up to roughly 20–30° of viewpoint
  change between two photos, so 9° leaves a wide margin, also for curved leaves and
  overlapping foliage.
- **Coverage.** Every surface needs to be seen from several angles. A part seen by no photo
  becomes a hole or a floater; a part seen from a single angle has unconstrained depth.
  Five rings cover the plant from below the leaves to the top.
- **Held-out test set.** Every 8th photo is held out for evaluation, so ~22 photos are never
  trained on. They are spread evenly because they come from every ring.
- **Budget.** 3DGS works well with 100–300 views. More than ~200 only lengthens COLMAP
  matching (exhaustive matching grows as N²) without improving the result.
- **No close-ups.** At 1.6 m the D7200 already resolves ~0.2 mm per pixel, and training at
  2000 px gives ~0.5 mm per pixel, enough for leaf edges and terracotta grain. Close-ups
  would need refocusing, which changes the lens intrinsics, and COLMAP assumes **one shared
  camera** for all photos.

---

## 3. Geometry: where to stand

Keep the **distance from the camera to the plant centre constant** (≈ 1.6 m for a plant
about 50 cm wide and 60 cm tall). Then the focus never needs to change and every photo has
the same scale.

**Distance for your plant.** Shoot in **portrait orientation** (the plant is taller than
wide, and the trailing vines hang down). Frame so that the whole plant and pot fill ~70 %
of the frame, with a margin on all sides.

| plant size (width × height, incl. pot and vines) | distance (portrait, 35 mm on DX) |
|---|---|
| 40 × 50 cm | ~1.3 m |
| 50 × 60 cm | **~1.6 m** |
| 60 × 80 cm | ~1.9 m |

**Rings.** At elevation angle θ and distance d, the camera is at a horizontal radius
r = d cos θ from the plant centre, and at height h = d sin θ above it. For d = 1.6 m:

| ring | θ | floor radius r | camera height above plant centre | tape marks |
|---|---|---|---|---|
| 1 | 0° | 1.60 m | 0 m | 40 marks, 25 cm apart |
| 2 | 20° | 1.50 m | 0.55 m | 40 marks, 24 cm apart |
| 3 | 40° | 1.23 m | 1.03 m | 40 marks, 19 cm apart |
| 4 | 60° | 0.80 m | 1.39 m | 32 marks, 16 cm apart |
| 5 | 80° | 0.28 m | 1.58 m | 12 positions around the plant |

Make the circles with a string of length r tied to a stick at the pot's centre. ±10 % on
r or the height is fine; what matters is a constant distance, smooth spacing and overlap.

```
side view                                   top view (rings 1–4)

            5  (80°)                              . . . . .
         4  (60°)                              .   . . .   .
      3  (40°)                                .  .  .-.  .  .
   2  (20°)                                   . .  ( P )  . .     P = plant
1 ──────────────── P   (0°, mid-pot)          .  .  '-'  .  .
                   pot                         .   . . .   .
──────────────────────── floor                    . . . . .
```

Rings 4–5 need the camera 1.4–1.6 m above the plant centre, which is easiest with the plant
on the floor. For the top ring, use the tripod's centre column tilted, or stand on a stool
and hold the camera carefully (higher shutter speed, ISO up to 400 is fine).

**Depth of field check.** At 1.6 m and 35 mm, standard depth of field covers ~1.3–2.0 m at
f/8 and ~1.25–2.25 m at f/11. Either covers a plant 50–60 cm deep when focused on its
centre. Training happens at about 1/3 of full resolution, so the standard
circle-of-confusion criterion is the right one. f/8 is preferred: past f/11 the D7200's
3.9 µm pixels start losing sharpness to diffraction.

---

## 4. Light, background and the scene

**Gaussian Splatting assumes a static scene under fixed lighting.** Each Gaussian learns a
colour (later, a view-dependent colour for the leaf sheen). It cannot explain light that
changes between photos, or leaves that move.

- **Light.** Use soft, constant light. An overcast day near a big window works, or two LED
  panels or lamps with diffusers at ~45° on either side. Avoid direct sun: it moves
  noticeably in 30–45 minutes and casts hard shadows. Avoid flash: it moves with the
  camera, so every photo would be lit differently. Turn off mixed light sources (e.g. warm
  bulbs plus daylight).
- **Stillness.** Any leaf movement creates blur and ghost Gaussians. Close windows, switch
  off fans and air conditioning, and don't walk too close to the vines.
- **Background.** Use a plain matte sheet in light grey or white, not green (the leaves)
  and not orange (the pot), so the foreground masks (rembg/SAM) separate the plant cleanly.
  Avoid glossy surfaces and mirrors.
- **Features for COLMAP.** A plain background has no features to match, which is why the
  textured newspaper or magazine goes on the floor around the pot. COLMAP uses it to find
  the poses; the masks then exclude it from training.
- **You.** Keep your shadow off the plant (stand on the side away from the light, or move
  the lights with you only if they're symmetric; better not to move them at all).

---

## 5. Shooting procedure (~45–60 min)

1. Set everything up as in the checklist, mark the rings, and measure white balance on a grey card.
2. Take one test photo. At 100 % zoom, check that the leaf edges are sharp and the
   highlights aren't clipped (blinkies). Adjust the shutter speed only, then **lock it**.
3. **Pilot:** shoot ring 1 only, ~20 photos every 18°. Upload them and wait for my check.
4. Full capture, ring by ring: 1 → 2 → 3 → 4 → 5, walking in **one direction** with
   consecutive positions. Then add extras where the foliage is complicated.
5. Don't delete anything in camera. Blurry or bad frames are filtered on the VM.
6. At the end, take one more photo from the first position of ring 1. Comparing it with
   the first photo shows whether the light or the plant changed during the session.

---

## 6. Uploading

Copy everything (JPEG + NEF) to the VM. At ~12 MB per JPEG and ~30 MB per NEF, ~200 photos
are about 8 GB, which fits easily on `/mnt`:

```bash
# from your laptop (replace <vm> with the VM's address)
rsync -avP /path/to/DCIM/1XXND7200/ ubuntu@<vm>:/mnt/data/plant/raw/
```

For the **pilot**, upload the ~20 ring-1 photos to `/mnt/data/plant/pilot/`. I'll then:
- run COLMAP on them;
- check that all are registered and the reprojection error is low;
- run rembg on a few to check that the masks separate plant and pot from the background.

That takes about 10 minutes and catches problems (wrong settings, moving leaves, a
background the masks can't separate) before you spend an hour on the full capture.

---

## 7. What goes wrong most often

| problem | symptom in the reconstruction | prevention |
|---|---|---|
| leaves moved (draft, touching) | blurry or doubled leaves, floaters | no airflow, don't brush the vines |
| auto white balance / exposure | colour flicker, splotchy leaves | M mode, fixed WB |
| refocused or zoomed | COLMAP fails or poses are inaccurate | MF after focusing once, tape the zoom ring |
| VR on while on a tripod | slight blur, small shifts of the optical centre | VR OFF |
| sun moved during capture | inconsistent shading, baked-in shadows | overcast or artificial light, finish within ~45 min |
| flash or your shadow | lighting differs per view | no flash, watch your shadow |
| too few high/low views | holes under leaves or on the top | all 5 rings |
| background same colour as the plant | bad masks | grey/white matte backdrop |
| plain background, nothing else | COLMAP can't register images | textured objects on the floor |

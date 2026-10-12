#!/usr/bin/env bash
# Stage 6: headless structure-from-motion with COLMAP (CPU build).
#   bash scripts/run_colmap.sh data/plant images_2000
# One shared OPENCV camera (fx, fy, cx, cy, k1, k2, p1, p2) for all images, exhaustive
# matching with guided matching, incremental mapping. Output: <scene>/colmap/{database.db,sparse/0}
set -euo pipefail
SCENE=${1:-data/plant}
IMAGES=${2:-images_2000}
OUT=$SCENE/colmap
mkdir -p $OUT/sparse
export QT_QPA_PLATFORM=offscreen            # headless: no X display
T=$(nproc)

colmap feature_extractor \
  --database_path $OUT/database.db --image_path $SCENE/$IMAGES \
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
  --SiftExtraction.use_gpu 0 --SiftExtraction.num_threads $T \
  --SiftExtraction.max_image_size 2000 --SiftExtraction.max_num_features 8192

colmap exhaustive_matcher \
  --database_path $OUT/database.db \
  --SiftMatching.use_gpu 0 --SiftMatching.num_threads $T --SiftMatching.guided_matching 1

colmap mapper \
  --database_path $OUT/database.db --image_path $SCENE/$IMAGES --output_path $OUT/sparse \
  --Mapper.num_threads $T --Mapper.ba_refine_principal_point 0

for m in $OUT/sparse/*/; do
  echo "=== model $m"; colmap model_analyzer --path $m
done

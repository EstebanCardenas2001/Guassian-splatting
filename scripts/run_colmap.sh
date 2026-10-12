#!/usr/bin/env bash
# Stage 6: structure-from-motion with COLMAP 4.2.1 built with CUDA (conda env "colmap").
#   bash scripts/run_colmap.sh data/plant images_2000 [global|incremental]
#
# * One shared OPENCV camera (fx, fy, cx, cy, k1, k2, p1, p2) for all images.
# * SIFT extraction and exhaustive matching on the GPU, with guided matching (re-matching
#   under the estimated two-view geometry, more inliers on repetitive leaves).
# * Mapper: "global" = COLMAP's built-in GLOMAP (global rotation averaging + global
#   positioning + bundle adjustment: all cameras solved at once, fast for many images);
#   "incremental" = classic COLMAP (adds images one by one; slower, kept as a cross-check).
# Output: <scene>/colmap_<mapper>/{database.db, sparse/0, timings.txt}
set -euo pipefail
SCENE=${1:-data/plant}
IMAGES=${2:-images_2000}
MAPPER=${3:-global}
OUT=$SCENE/colmap_$MAPPER
mkdir -p $OUT/sparse
source /mnt/miniforge3/etc/profile.d/conda.sh
set +u; conda activate colmap; set -u
export QT_QPA_PLATFORM=offscreen            # headless: no X display
T=$(nproc)
: > $OUT/timings.txt
step() { local name=$1; shift; local t0=$(date +%s.%N); "$@"; \
         echo "$name $(echo "$(date +%s.%N) - $t0" | bc) s" | tee -a $OUT/timings.txt; }

step extract colmap feature_extractor \
  --database_path $OUT/database.db --image_path $SCENE/$IMAGES \
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
  --FeatureExtraction.use_gpu 1 --FeatureExtraction.max_image_size 2000 \
  --SiftExtraction.max_num_features 8192

step match colmap exhaustive_matcher \
  --database_path $OUT/database.db \
  --FeatureMatching.use_gpu 1 --FeatureMatching.guided_matching 1

if [ "$MAPPER" = global ]; then
  step map colmap global_mapper \
    --database_path $OUT/database.db --image_path $SCENE/$IMAGES --output_path $OUT/sparse
else
  step map colmap mapper \
    --database_path $OUT/database.db --image_path $SCENE/$IMAGES --output_path $OUT/sparse \
    --Mapper.num_threads $T --Mapper.ba_refine_principal_point 0
fi

for m in $OUT/sparse/*/; do
  echo "=== model $m"; colmap model_analyzer --path $m
done

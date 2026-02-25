# Path to your template config (refref.yaml)
TEMPLATE_CFG="configs/shape/nerf/refref.yaml"

# Parent directory that contains all scene folders (ball_hdr, cat_hdr, cube_hdr, ...)
DATASET_ROOT="/home/projects/RefRef/image_data/env_map_scene/single-convex"

# for scene_dir in "$DATASET_ROOT"/*/; do
for scene_dir in $(printf '%s\n' "$DATASET_ROOT"/*/ | sort -r); do
  # scene_dir looks like /.../single-convex/ball_hdr/
  scene_name="$(basename "$scene_dir")"
  
  # Where to write per-scene configs
  OUT_CFG_DIR="data/model/${scene_name}"
  mkdir -p "$OUT_CFG_DIR"
  cfg_out="$OUT_CFG_DIR/${scene_name}.yaml"

  # Create a per-scene yaml by overriding name and dataset_dir in the template
  sed \
    -e "s/^name:.*/name: ${scene_name}/" \
    -e "s|^dataset_dir:.*|dataset_dir: ${DATASET_ROOT}|" \
    "$TEMPLATE_CFG" > "$cfg_out"

  echo ""
  echo "Running training for scene: ${scene_name}"
  echo ""
  python run_training.py --cfg "$cfg_out"
done

# -e "s/^  sphere_direction:.*/  sphere_direction: True  # infinity far light only/" \
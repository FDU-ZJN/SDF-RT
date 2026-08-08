#!/usr/bin/env bash
# Run this script on the KV260 to validate the deployed SDF-RT overlay.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "${EUID}" -ne 0 ]]; then
    echo "KV260 verification must run as root because PYNQ maps PS/PL registers." >&2
    exit 77
fi

width="${1:-${KV260_RT_WIDTH:-320}}"
height="${2:-${KV260_RT_HEIGHT:-180}}"
batch_rays="${3:-${KV260_RT_BATCH_RAYS:-65536}}"
frames="${4:-${KV260_RT_FRAMES:-1}}"
pynq_python="${PYNQ_PYTHON:-/usr/local/share/pynq-venv/bin/python}"
pynq_env="${PYNQ_ENV:-/etc/profile.d/pynq_venv.sh}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${script_dir}/verification/${run_id}"
results="${output_dir}/results.npz"
image="${output_dir}/render.png"
depth_image="${output_dir}/depth.png"
bitstream="${output_dir}/KV260_verify.bit"
hwh="${output_dir}/KV260_verify.hwh"
pipeline_args=()
if [[ "${5:-}" == "--no-pipeline" || "${KV260_RT_NO_PIPELINE:-0}" == "1" ]]; then
    pipeline_args+=(--no-pipeline)
fi
[[ "$#" -le 5 ]] || { echo "usage: verify_render.sh [width height batch_rays frames [--no-pipeline]]" >&2; exit 2; }

for required in KV260.bit KV260.hwh pynq.py sdf_rt/__init__.py sdf_rt/cli.py \
    mem/sdf_global_mem.mem mem/sdf_local_mem.mem \
    mem/sdf_local_mapping.mem mem/triangle_ddr.bin mem/pynq_render_data.bin; do
    [[ -f "${script_dir}/${required}" ]] || {
        echo "Missing deployment asset: ${script_dir}/${required}" >&2
        exit 2
    }
done

[[ "${width}" =~ ^[1-9][0-9]*$ ]] || { echo "KV260_RT_WIDTH must be positive" >&2; exit 2; }
[[ "${height}" =~ ^[1-9][0-9]*$ ]] || { echo "KV260_RT_HEIGHT must be positive" >&2; exit 2; }
[[ "${batch_rays}" =~ ^[1-9][0-9]*$ ]] || { echo "KV260_RT_BATCH_RAYS must be positive" >&2; exit 2; }
[[ "${frames}" =~ ^[1-9][0-9]*$ ]] || { echo "KV260_RT_FRAMES must be positive" >&2; exit 2; }
[[ "${batch_rays}" -le 65536 ]] || { echo "KV260_RT_BATCH_RAYS exceeds pynq.py capacity" >&2; exit 2; }
[[ -x "${pynq_python}" ]] || { echo "PYNQ Python is not executable: ${pynq_python}" >&2; exit 2; }
[[ -r "${pynq_env}" ]] || { echo "PYNQ environment script is missing: ${pynq_env}" >&2; exit 2; }

# Non-interactive SSH shells do not source the board's PYNQ/XRT setup.
source "${pynq_env}"

mkdir -p "${output_dir}"
cd "${script_dir}"
cp --reflink=auto KV260.bit "${bitstream}"
cp KV260.hwh "${hwh}"

echo "[VERIFY] KV260 SDF-RT ${width}x${height}, batch=${batch_rays}, frames=${frames}"
"${pynq_python}" pynq.py \
    --bitstream "${bitstream}" \
    --hwh "${hwh}" \
    --width "${width}" \
    --height "${height}" \
    --batch-rays "${batch_rays}" \
    --frames "${frames}" \
    --sdf-timeout 30 \
    --setup-timeout 5 \
    --batch-timeout 60 \
    --results "${results}" \
    --output "${image}" \
    --depth-output "${depth_image}" \
    "${pipeline_args[@]}"

if [[ "${frames}" -eq 1 ]]; then
    verify_image="${image}"
else
    last_frame="$(printf '%03d' "$((frames - 1))")"
    verify_image="${output_dir}/render_frame_${last_frame}.png"
fi

"${pynq_python}" - "${results}" "${verify_image}" "${depth_image}" "${width}" "${height}" <<'PY'
import sys

import numpy as np
from PIL import Image

results_path, image_path, depth_path, width_arg, height_arg = sys.argv[1:]
width = int(width_arg)
height = int(height_arg)
expected_pixels = width * height

with np.load(results_path) as results:
    required = {"width", "height", "hit", "hit_id", "hit_t_bits"}
    missing = required.difference(results.files)
    if missing:
        raise SystemExit(f"[VERIFY] result archive misses keys: {sorted(missing)}")
    if int(results["width"]) != width or int(results["height"]) != height:
        raise SystemExit("[VERIFY] result archive dimensions do not match the request")
    hit = np.asarray(results["hit"], dtype=bool)
    hit_t = np.asarray(results["hit_t_bits"], dtype=np.uint32).view(np.float32)
    if hit.shape != (expected_pixels,) or hit_t.shape != (expected_pixels,):
        raise SystemExit("[VERIFY] result array shape does not match the requested frame")
    hit_count = int(np.count_nonzero(hit))
    if hit_count == 0:
        raise SystemExit("[VERIFY] all rays missed; SDF traversal or camera setup is invalid")
    if not np.all(np.isfinite(hit_t[hit])) or not np.all(hit_t[hit] >= 0.0):
        raise SystemExit("[VERIFY] hit rays contain invalid distances")

for path in (image_path, depth_path):
    with Image.open(path) as image:
        if image.size != (width, height):
            raise SystemExit(f"[VERIFY] {path} has size {image.size}, expected {(width, height)}")

print(
    f"[VERIFY] PASS: pixels={expected_pixels} hits={hit_count} "
    f"({100.0 * hit_count / expected_pixels:.2f}%), "
    f"hit_t=[{float(hit_t[hit].min()):.6f}, {float(hit_t[hit].max()):.6f}]"
)
PY

echo "[VERIFY] artifacts: ${output_dir}"

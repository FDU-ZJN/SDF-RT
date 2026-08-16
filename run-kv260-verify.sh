#!/usr/bin/env bash
# Invoke the mounted KV260 deployment's verification script through SSH.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
remote_dir="${KV260_RT_REMOTE_DIR:-kv260_rt}"
sudo_password="${KV260_SUDO_PASSWORD:-xilinx123}"
render_threads="${KV260_RT_RENDER_THREADS:-4}"
display_hold_seconds="${KV260_RT_DISPLAY_HOLD_SECONDS:-3}"
validate_result_stream="${KV260_RT_VALIDATE_RESULT_STREAM:-0}"
rotation_period_frames="${KV260_RT_ROTATION_PERIOD_FRAMES:-180}"
grid_min="${KV260_RT_GRID_MIN:-}"
grid_max="${KV260_RT_GRID_MAX:-}"
printf -v remote_command \
    "cd %q && exec sudo -S -p '' -E env KV260_RT_RENDER_THREADS=%q KV260_RT_DISPLAY_HOLD_SECONDS=%q KV260_RT_VALIDATE_RESULT_STREAM=%q KV260_RT_ROTATION_PERIOD_FRAMES=%q KV260_RT_GRID_MIN=%q KV260_RT_GRID_MAX=%q ./verify_render.sh" \
    "${remote_dir}" "${render_threads}" "${display_hold_seconds}" "${validate_result_stream}" "${rotation_period_frames}" "${grid_min}" "${grid_max}"

[[ -x "${script_dir}/kv260_rt/verify_render.sh" ]] || {
    echo "Board verification script is missing or not executable: ${script_dir}/kv260_rt/verify_render.sh" >&2
    exit 2
}

printf '%s\n' "${sudo_password}" |
    ssh -tt kv260 "${remote_command}" "$@"

#!/usr/bin/env bash
# Invoke the mounted KV260 deployment's verification script through SSH.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
remote_dir="${KV260_RT_REMOTE_DIR:-kv260_rt}"
sudo_password="${KV260_SUDO_PASSWORD:-xilinx123}"
render_threads="${KV260_RT_RENDER_THREADS:-4}"
save_ppm="${KV260_RT_SAVE_PPM:-}"
display_hold_seconds="${KV260_RT_DISPLAY_HOLD_SECONDS:-3}"
printf -v remote_command \
    "cd %q && exec sudo -S -p '' -E env KV260_RT_RENDER_THREADS=%q KV260_RT_SAVE_PPM=%q KV260_RT_DISPLAY_HOLD_SECONDS=%q ./verify_render.sh" \
    "${remote_dir}" "${render_threads}" "${save_ppm}" "${display_hold_seconds}"

[[ -x "${script_dir}/kv260_rt/verify_render.sh" ]] || {
    echo "Board verification script is missing or not executable: ${script_dir}/kv260_rt/verify_render.sh" >&2
    exit 2
}

printf '%s\n' "${sudo_password}" |
    ssh -tt kv260 "${remote_command}" "$@"

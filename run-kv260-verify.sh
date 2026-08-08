#!/usr/bin/env bash
# Invoke the mounted KV260 deployment's verification script through SSH.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
remote_dir="${KV260_RT_REMOTE_DIR:-kv260_rt}"
sudo_password="${KV260_SUDO_PASSWORD:-xilinx123}"

[[ -x "${script_dir}/kv260_rt/verify_render.sh" ]] || {
    echo "Board verification script is missing or not executable: ${script_dir}/kv260_rt/verify_render.sh" >&2
    exit 2
}

printf '%s\n' "${sudo_password}" |
    ssh -tt kv260 "cd '${remote_dir}' && exec sudo -S -p '' -E ./verify_render.sh" "$@"

#!/usr/bin/env bash
# Mount apple.top:Workspace (relative to the remote home directory) at ./workspace.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
mount_point="${script_dir}/kv260_rt"
remote_path="kv260:kv260_rt"
ssh_config="${HOME}/.ssh/config"

is_mounted() {
  mountpoint -q "${mount_point}"
}

mount_workspace() {
  command -v sshfs >/dev/null || {
    echo "sshfs is not installed." >&2
    exit 1
  }

  [ -f "${ssh_config}" ] || {
    echo "SSH config not found: ${ssh_config}" >&2
    exit 1
  }

  mkdir -p "${mount_point}"

  if is_mounted; then
    echo "Already mounted: ${mount_point}"
    return
  fi

  if [ -n "$(find "${mount_point}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    echo "Refusing to mount over a non-empty directory: ${mount_point}" >&2
    exit 1
  fi

  sshfs -F "${ssh_config}" \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
    "${remote_path}" "${mount_point}"

  echo "Mounted ${remote_path} at ${mount_point}"
}

unmount_workspace() {
  if ! is_mounted; then
    echo "Not mounted: ${mount_point}"
    return
  fi

  if command -v fusermount3 >/dev/null; then
    fusermount3 -u "${mount_point}"
  else
    umount "${mount_point}"
  fi

  echo "Unmounted ${mount_point}"
}

case "${1:-mount}" in
  mount) mount_workspace ;;
  unmount) unmount_workspace ;;
  status)
    if is_mounted; then
      echo "Mounted: ${mount_point}"
    else
      echo "Not mounted: ${mount_point}"
    fi
    ;;
  *)
    echo "Usage: $0 [mount|unmount|status]" >&2
    exit 2
    ;;
esac

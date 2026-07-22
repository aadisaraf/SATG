#!/usr/bin/env bash
# =============================================================================
# node_boot_recover.sh — heal a low-priority Azure ML node and resume training
# =============================================================================
# Runs at boot (satg-recover.service) AND every 5 min (satg-watchdog.timer), so
# both preemptions and mid-uptime crashes self-heal. Idempotent: if everything
# is healthy and training is running, it does nothing.
#
# Order matters:
#   1. temp-disk dirs + NVMe (ephemeral, recreated after every preemption)
#   2. blob mount — blobfuse2 refuses to mount if EITHER the file_cache dir or
#      the mountpoint is non-empty, so both are cleared first
#   3. HARD verify it is a real FUSE mount before touching anything under it
#      (an unmounted ~/blob is a plain local dir; writing into it poisons the
#      mountpoint and blocks all future mounts)
#   4. symlinks, then relaunch the task in ~/.satg_task if not already running
#
# NOTE: the systemd units MUST use KillMode=process, otherwise systemd SIGKILLs
# the blobfuse daemon and tmux session this script spawns as soon as it exits.
#
# ~/.satg_task format:  "<tmux-session>|<command run from ~/SATG>"
# =============================================================================
set -uo pipefail
LOG="$HOME/satg_boot.log"
exec >> "$LOG" 2>&1
echo "=== $(date -u) boot recovery ==="

# 1. temp-disk dirs (wiped on every preemption; /mnt is root-owned)
sudo mkdir -p /mnt/blobcache /mnt/gta5_zips /mnt/tmp
sudo chown "$USER:$USER" /mnt/blobcache /mnt/gta5_zips /mnt/tmp

# 1b. fast local NVMe (ephemeral — reformat+mount after a preemption)
if [ -b /dev/nvme0n1 ] && ! mountpoint -q /nvme; then
    sudo mkdir -p /nvme
    sudo mount /dev/nvme0n1 /nvme 2>/dev/null \
        || { sudo mkfs.ext4 -F /dev/nvme0n1 && sudo mount /dev/nvme0n1 /nvme; }
    sudo chown "$USER:$USER" /nvme
    echo "nvme mounted at /nvme"
fi
[ -d /nvme ] && mkdir -p /nvme/blobcache

# 2. blob mount (only if not already a healthy FUSE mount)
if findmnt "$HOME/blob" >/dev/null 2>&1 && ls "$HOME/blob" >/dev/null 2>&1; then
    echo "blob mounted and healthy"
else
    echo "blob missing or stale — remounting"
    fusermount -u "$HOME/blob" 2>/dev/null || true
    sudo umount -l "$HOME/blob" 2>/dev/null || true
    # blobfuse2 requires an EMPTY file_cache dir
    rm -rf /nvme/blobcache/* /nvme/blobcache/.[!.]* 2>/dev/null || true
    # ...and an EMPTY mountpoint. Only prune empty dirs, and only when we have
    # confirmed nothing is mounted there — never risk deleting real blob data.
    mkdir -p "$HOME/blob"
    if ! findmnt "$HOME/blob" >/dev/null 2>&1; then
        find "$HOME/blob" -mindepth 1 -depth -type d -empty -delete 2>/dev/null || true
    fi
    blobfuse2 mount "$HOME/blob" --config-file="$HOME/blobfuse2.yaml" && echo "blob remounted"
fi

# 3. HARD verify: must be a real FUSE mount and readable, else stop here.
if ! findmnt "$HOME/blob" >/dev/null 2>&1 || ! ls "$HOME/blob" >/dev/null 2>&1; then
    echo "blob unavailable (not a live mount) — NOT relaunching training"
    exit 1
fi

# 4. symlinks — safe now that blob is verified mounted. data/ is a Python
# package, so only its dataset SUBDIRS are linked, never data/ itself.
if [ -d "$HOME/SATG/data" ]; then
    mkdir -p "$HOME/blob/data/GTA5" "$HOME/blob/data/cityscapes" \
             "$HOME/blob/checkpoints" "$HOME/blob/logs"
    ln -sfn "$HOME/blob/data/GTA5"       "$HOME/SATG/data/GTA5"
    ln -sfn "$HOME/blob/data/cityscapes" "$HOME/SATG/data/cityscapes"
    ln -sfn "$HOME/blob/checkpoints"     "$HOME/SATG/checkpoints"
    ln -sfn "$HOME/blob/logs"            "$HOME/SATG/cloud/logs"
    echo "symlinks ensured"
fi

# 5. relaunch the recorded task, only if that tmux session isn't already up
TASK="$HOME/.satg_task"
if [ -f "$TASK" ]; then
    sess="$(cut -d'|' -f1 "$TASK")"
    cmd="$(cut -d'|' -f2- "$TASK")"
    if [ -z "$sess" ] || [ -z "$cmd" ]; then
        echo "~/.satg_task malformed — skipping relaunch"
    elif tmux has-session -t "$sess" 2>/dev/null; then
        echo "task [$sess] already running — nothing to do"
    else
        cd "$HOME/SATG" && tmux new -d -s "$sess" "$cmd" && echo "relaunched [$sess]: $cmd"
    fi
else
    echo "no ~/.satg_task — nothing to resume"
fi
echo "recovery done."

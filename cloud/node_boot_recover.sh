#!/usr/bin/env bash
# =============================================================================
# node_boot_recover.sh — auto-recover a low-priority Azure ML node after a
# preemption/reboot, then resume the active task.
# =============================================================================
# Low-priority cluster nodes get preempted; on reboot the temp disk (/mnt) and
# tmux are wiped and the blob mount drops, but $HOME survives. This script:
#   1. recreates the temp-disk dirs (/mnt/blobcache, /mnt/gta5_zips),
#   2. remounts blob from ~/blobfuse2.yaml,
#   3. relaunches the task recorded in ~/.satg_task inside a detached tmux
#      session — only if that session isn't already running.
#
# ~/.satg_task format:  "<tmux-session>|<command run from ~/SATG>"
#   echo 'gta5|PAR=4 bash cloud/download_gta5_parallel.sh' > ~/.satg_task
#   echo 'satg-train|bash cloud/run_phase9.sh'            > ~/.satg_task
#
# Run manually any time (also does the recovery immediately):
#   bash cloud/node_boot_recover.sh
#
# Install as a boot service so preemptions self-heal (see runbook Step 2b).
# =============================================================================
set -uo pipefail
LOG="$HOME/satg_boot.log"
exec >> "$LOG" 2>&1
echo "=== $(date -u) boot recovery ==="

# 1. temp-disk dirs (wiped on every preemption; /mnt is root-owned). /mnt/tmp is
# the user-owned scratch used as TMPDIR (Cityscapes unzip, extraction temps).
sudo mkdir -p /mnt/blobcache /mnt/gta5_zips /mnt/tmp
sudo chown "$USER:$USER" /mnt/blobcache /mnt/gta5_zips /mnt/tmp

# 1b. fast local NVMe (ephemeral — reformat+mount after a preemption). Holds the
# blobfuse read-cache so training reads are local NVMe, not over the network.
if [ -b /dev/nvme0n1 ] && ! mountpoint -q /nvme; then
    sudo mkdir -p /nvme
    sudo mount /dev/nvme0n1 /nvme 2>/dev/null \
        || { sudo mkfs.ext4 -F /dev/nvme0n1 && sudo mount /dev/nvme0n1 /nvme; }
    sudo chown "$USER:$USER" /nvme
    echo "nvme mounted at /nvme"
fi
[ -d /nvme ] && mkdir -p /nvme/blobcache

# 2. remount blob if not mounted OR stale. After a preemption the mount can go
# stale: findmnt still lists it but I/O fails with "Transport endpoint is not
# connected". Test with a real listing, and force-unmount before remounting.
if findmnt "$HOME/blob" >/dev/null 2>&1 && ls "$HOME/blob" >/dev/null 2>&1; then
    echo "blob mounted and healthy"
else
    echo "blob missing or stale — remounting"
    fusermount -u "$HOME/blob" 2>/dev/null || true
    sudo umount -l "$HOME/blob" 2>/dev/null || true
    # blobfuse2 REFUSES to mount if its file_cache dir is non-empty
    # ("temp directory not empty"), so clear it before every mount attempt.
    rm -rf /nvme/blobcache/* /nvme/blobcache/.[!.]* 2>/dev/null || true
    blobfuse2 mount "$HOME/blob" --config-file="$HOME/blobfuse2.yaml" && echo "blob remounted"
fi

# Never launch training against a dead mount — it would just crash instantly.
if ! ls "$HOME/blob" >/dev/null 2>&1; then
    echo "blob still unavailable — NOT relaunching training"
    exit 1
fi

# 2b. re-establish dataset/output symlinks (data/ is a Python package, so only
# its GTA5/ and cityscapes/ SUBDIRS are symlinked onto blob — never data/ itself).
if [ -d "$HOME/SATG/data" ]; then
    mkdir -p "$HOME/blob/data/GTA5" "$HOME/blob/data/cityscapes"
    ln -sfn "$HOME/blob/data/GTA5"       "$HOME/SATG/data/GTA5"
    ln -sfn "$HOME/blob/data/cityscapes" "$HOME/SATG/data/cityscapes"
    ln -sfn "$HOME/blob/checkpoints"     "$HOME/SATG/checkpoints"
    ln -sfn "$HOME/blob/logs"            "$HOME/SATG/cloud/logs"
    echo "symlinks ensured"
fi

# 2c. (pre-warm removed) A 16-way parallel scan of 164 GB right after mount was
# crashing the blobfuse daemon ("Transport endpoint is not connected"), killing
# training. Training's own gentle 8-worker reads warm the NVMe cache lazily
# instead — slower first epoch, but the mount stays alive.

# 3. relaunch the recorded task, once
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

# SATG — Azure ML Compute Cluster + Blob Runbook (Windows / PowerShell)

Definitive end-to-end runbook for training **Structure-Aware Trust Gating (SATG)**
on an Azure ML **compute cluster** node, driven from **Windows / PowerShell**, with
all datasets + checkpoints on **Azure Blob** so they survive node recycling. Every
fix discovered during bring-up is baked in.

> **Shell tags:** 🖥️ = PowerShell on your laptop · ☁️ = bash on the node (after SSH).

---

## Storage you need

| Component | Size |
|---|---|
| GTA5 (images + trainId labels) | ~58 GB |
| Cityscapes (leftImg8bit + gtFine + trainIds) | ~13 GB |
| SATG heatmaps (2,975 × 32 MB) | ~95 GB |
| Checkpoints (~41 runs × ~1 GB) | ~45 GB |
| Logs + visualizations | ~2 GB |
| **Total persistent (blob)** | **~210 GB — provision ~250 GB** |

Plus **~50 GB free on the local temp disk `/mnt`** for GTA5 zip staging, and ~40 GB
of `/mnt` for the blob read-cache during training.

---

## 0. Node facts + keep it alive

| Field | Value |
|-------|-------|
| IP | `172.182.230.39` |
| SSH port | `50000` |
| User | `azureuser` |
| SSH key | `~\.ssh\id_rsa_azureml` |
| Resource group | `pranav-rg` · Region `westus3` |
| Blob storage acct | `satg131168` · container `satg-data` (reused — GTA5 resumes) |

🖥️ **PowerShell (local)** — set once per window:

```powershell
$SATG_HOST="172.182.230.39"; $SATG_PORT="50000"; $SATG_USER="azureuser"
$SATG_KEY="$HOME\.ssh\id_rsa_azureml"
$SA="satg131168"                      # existing storage account
```

**⚠️ FIRST, in Azure ML Studio: set the cluster's Minimum nodes = 1.**
Compute → Compute clusters → *(this node's cluster)* → Edit → **Minimum number of
nodes = 1** → Save. Without this the cluster deallocates the node when it looks idle
(a manual SSH job doesn't count), wiping `/mnt` and killing your run. Set back to 0
when the whole project is done.

---

## 1. Connect (with keepalive so SSH doesn't drop)

🖥️ **PowerShell (local):**

```powershell
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=5 -i $SATG_KEY -p $SATG_PORT "$($SATG_USER)@$($SATG_HOST)"
```

Password prompt instead of a login = wrong key path; fix `$SATG_KEY`.

---

## 2. Mount blob (small cache for the data-prep phase)

☁️ **on the node** — install blobfuse2 if missing, make a cache dir on the temp
disk, write the config, mount. Use a **small 4 GB cache during data prep** so `/mnt`
has room for GTA5 staging (we bump it up before training in Step 7).

```bash
command -v blobfuse2 >/dev/null || {
  wget -q "https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb" -O /tmp/pmp.deb
  sudo dpkg -i /tmp/pmp.deb && sudo apt-get update && sudo apt-get install -y blobfuse2
}
sudo mkdir -p /mnt/blobcache && sudo chown "$USER:$USER" /mnt/blobcache

cat > ~/blobfuse2.yaml <<'YAML'
components: [libfuse, file_cache, attr_cache, azstorage]
libfuse: {attribute-expiration-sec: 120, entry-expiration-sec: 120}
file_cache: {path: /mnt/blobcache, timeout-sec: 600, max-size-mb: 4000}
attr_cache: {timeout-sec: 120}
azstorage:
  type: block
  mode: key
  account-name: satg131168
  account-key: PASTE_KEY_HERE
  endpoint: https://satg131168.blob.core.windows.net
  container: satg-data
YAML
nano ~/blobfuse2.yaml          # replace PASTE_KEY_HERE with your storage key

mkdir -p ~/blob
blobfuse2 mount ~/blob --config-file="$HOME/blobfuse2.yaml"
findmnt ~/blob                 # MUST print a 'blobfuse2 fuse' line
mkdir -p ~/blob/data ~/blob/checkpoints ~/blob/logs
```

> Get the key from 🖥️ PowerShell if you don't have it:
> `az storage account keys list -g pranav-rg -n satg131168 --query "[0].value" -o tsv`

---

## 3. Code + link to blob

☁️ **on the node** — clone locally (cheap to re-clone after a recycle), symlink the
heavy dirs onto blob:

```bash
cd ~ && git clone -b azure-ml-runbook https://github.com/aadisaraf/SATG.git SATG && cd ~/SATG
rm -rf data checkpoints cloud/logs
ln -sfn ~/blob/data data
ln -sfn ~/blob/checkpoints checkpoints
ln -sfn ~/blob/logs cloud/logs
ls -la data checkpoints cloud/logs      # each -> /home/azureuser/blob/...
```

Private repo? Clone with a token:
`git clone -b azure-ml-runbook https://<user>:<PAT>@github.com/aadisaraf/SATG.git SATG`

---

## 4. Python + environment

☁️ **on the node** — this node has `python3` but no bare `python`; the scripts call
`python`/`pip`, so alias them, then run setup:

```bash
sudo ln -sf "$(command -v python3)" /usr/local/bin/python
sudo ln -sf "$(command -v pip3)"    /usr/local/bin/pip
cd ~/SATG && bash cloud/setup.sh        # installs deps, verifies CUDA -> ">>> GPU READY <<<"
```

---

## 5. Kaggle credentials (for the two Cityscapes datasets)

🖥️ **PowerShell (local)** — upload your token (from kaggle.com → Settings → Create
New API Token):

```powershell
scp -i $SATG_KEY -P $SATG_PORT "$HOME\Downloads\kaggle.json" "$($SATG_USER)@$($SATG_HOST):~/kaggle.json"
```

☁️ **on the node:**

```bash
mkdir -p ~/.kaggle && mv ~/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
```

---

## 6. Download GTA5 — parallel, local-staged (fast)

TU Darmstadt throttles each connection to ~1.3 MB/s (sequential ≈ 14 h). The
parallel downloader fetches parts concurrently, **staging zips on local `/mnt`**
(not through blobfuse — writing big files straight to the mount fails with curl-23)
and moving only extracted PNGs to blob. ☁️ **on the node:**

```bash
sudo mkdir -p /mnt/gta5_zips && sudo chown "$USER" /mnt/gta5_zips
cd ~/SATG
tmux new -s gta5
PAR=4 bash cloud/download_gta5_parallel.sh
# Ctrl-B D to detach; tmux attach -t gta5 to check
```

Expect 4 parts at once, ~4–5 MB/s aggregate → **~3–4 h**. Prints `[NN] done` per part
(10 total), then `=== GTA5 download complete ===`. Resume-safe: re-run to continue.
Monitor from another shell:

```bash
echo "images: $(find ~/blob/data/GTA5/images -name '*.png' | wc -l) / 24966"
df -h /mnt      # keep well under full; if it climbs, use PAR=3
```

---

## 7. Cityscapes + heatmaps, then grow the cache

☁️ **on the node** — finish data prep (skips the now-present GTA5 download; does GTA5
label→trainId mapping, your 2 Kaggle Cityscapes datasets, and the 95 GB of heatmaps):

```bash
cd ~/SATG
export TMPDIR=/mnt
tmux new -s satg-data
bash cloud/prepare_data.sh
# wait for: >>> CITYSCAPES READY <<<   and   >>> HEATMAP VALIDATION PASSED <<<
```

Then **bump the blob cache back up for training read speed** (GTA5 staging is done,
so `/mnt` is free):

```bash
sudo rm -rf /mnt/gta5_zips
sed -i 's/max-size-mb: 4000/max-size-mb: 40000/' ~/blobfuse2.yaml
fusermount -u ~/blob; blobfuse2 mount ~/blob --config-file="$HOME/blobfuse2.yaml"
findmnt ~/blob
```

---

## 8. Train (eviction/recycle-safe)

☁️ **on the node:**

```bash
wandb login                       # or: wandb offline
cd ~/SATG
tmux new -s satg-train
bash cloud/run_phase8.sh          # 6 baselines
bash cloud/run_phase9.sh          # ~35 SATG + ablation runs
```

Detach Ctrl-B D. **After any recycle/drop**, reconnect and re-run the *same* command:
finished runs (`checkpoints/<run>/.done`) skip, the interrupted one resumes from
`last.pth` via `--resume`, the rest run fresh. Recovery if the node was wiped:

```bash
blobfuse2 mount ~/blob --config-file=$HOME/blobfuse2.yaml    # if unmounted
cd ~/SATG || (cd ~ && git clone -b azure-ml-runbook https://github.com/aadisaraf/SATG.git SATG && cd SATG && \
  ln -sfn ~/blob/data data && ln -sfn ~/blob/checkpoints checkpoints && ln -sfn ~/blob/logs cloud/logs)
bash cloud/run_phase9.sh
```

Single run / ablation:
```bash
python -m training.trainer --config configs/satg_hard.yaml --resume seed=42
python -m training.trainer --config configs/satg_hard.yaml \
    --run_name grid_tc0.95_ts0.70_seed42 \
    trust_gate.tau_conf=0.95 trust_gate.tau_struct=0.70 seed=42
```

> ⚠️ Training reads GTA5 + heatmaps (~165 GB working set) from blob through a ~40 GB
> cache, so cache-miss reads hit the network — expect it slower than local-disk time
> estimates. If too slow: bigger-temp-disk SKU (larger cache) or trim scope.

---

## 9. Monitor + analysis

☁️ **on the node:**

```bash
tail -f cloud/logs/phase8_source_only_seed42.log
grep "★ New best" cloud/logs/*.log | tail
watch -n2 nvidia-smi
```

After training — Phase 10 figures/tables:
```bash
for v in satg_hard satg_soft_weight satg_soft_label; do
  python -m visualization.visualize --checkpoint checkpoints/${v}_seed42/best.pth \
      --config configs/${v}.yaml --num_images 10 --output_dir visualizations/
done
# coverage plots + per-experiment mIoU mean±std: see snippets committed in this file's history
pytest -q          # optional: validate the code
```

---

## 10. Results + shutdown

🖥️ **PowerShell (local)** — pull from blob (durable, node not required):

```powershell
$KEY = az storage account keys list -g pranav-rg -n $SA --query "[0].value" -o tsv
az storage blob download-batch --account-name $SA --account-key $KEY -s satg-data/checkpoints -d .\checkpoints
az storage blob download-batch --account-name $SA --account-key $KEY -s satg-data/logs        -d .\cloud\logs
```

Then set the cluster **min nodes back to 0** (Studio) to stop billing. Blob data stays.

---

## 11. Troubleshooting

| Symptom | Fix |
|--------|-----|
| `export`/bash fails in PowerShell | That's a ☁️ node command; run it inside SSH. Local vars: `$X="y"`. |
| SSH `Connection reset` mid-run | Reconnect with the keepalive flags (Step 1). Work in `tmux` so it survives. |
| Node gone / `tmux ls` empty / `~/blob` missing after a drop | Cluster recycled the node — **set min nodes = 1**. Remount blob + re-clone (Step 8 recovery). Blob data is safe. |
| `curl (23) Failure writing` on GTA5 | You wrote zips straight to blob. Use `download_gta5_parallel.sh` (stages on `/mnt`), and keep the cache at 4 GB during download. |
| `python: command not found` | Run the `ln -sf` aliases in Step 4. |
| blobfuse mount fails | `blobfuse2 mount ~/blob --config-file=$HOME/blobfuse2.yaml --foreground` to see the error; check account/key/container and `/mnt/blobcache`. |
| `/mnt` fills during download | Lower to `PAR=3`; ensure cache is 4 GB (`grep max-size-mb ~/blobfuse2.yaml`). |
| CUDA OOM | Append `training.batch_size=2 training.crop_size="[384,384]"`. |

---

## Appendix — one-shot bring-up (after mount + creds are set)

```bash
# fresh node, blob already mounted at ~/blob with creds in ~/blobfuse2.yaml
sudo ln -sf "$(command -v python3)" /usr/local/bin/python
sudo ln -sf "$(command -v pip3)"    /usr/local/bin/pip
cd ~ && git clone -b azure-ml-runbook https://github.com/aadisaraf/SATG.git SATG && cd ~/SATG
rm -rf data checkpoints cloud/logs
ln -sfn ~/blob/data data && ln -sfn ~/blob/checkpoints checkpoints && ln -sfn ~/blob/logs cloud/logs
bash cloud/setup.sh
mkdir -p ~/.kaggle && chmod 600 ~/.kaggle/kaggle.json     # after scp'ing token
sudo mkdir -p /mnt/gta5_zips && sudo chown "$USER" /mnt/gta5_zips
tmux new -s gta5   # then: PAR=4 bash cloud/download_gta5_parallel.sh
# after GTA5: bump cache to 40000 + remount, then: bash cloud/prepare_data.sh
# then: wandb login && bash cloud/run_phase8.sh && bash cloud/run_phase9.sh
```

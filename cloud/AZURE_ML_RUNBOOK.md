# SATG — Azure ML Compute Cluster + Blob Storage Runbook (Windows / PowerShell)

End-to-end runbook for training **Structure-Aware Trust Gating (SATG)** on the
`chesstraining` **Azure ML compute cluster**, driven from a **Windows / PowerShell**
laptop, with **all datasets + checkpoints on Azure Blob storage** so they survive
node recycling.

> **Read the shell tag on every code block:**
> - 🖥️ **PowerShell (local)** — runs on your Windows laptop.
> - ☁️ **Bash (on the node)** — runs on the cluster node *after* you SSH in.

---

## Architecture (why it's built this way)

A compute **cluster** node is transient: when the cluster scales down it is
**deallocated and its local disks are wiped**. It also can't take an attached data
disk. So:

- **Persistent stuff → Azure Blob** (mounted at `~/blob` via blobfuse2): the repo's
  `data/`, `checkpoints/`, and `cloud/logs/` are **symlinked onto the blob mount**.
  They survive any node recycle.
- **Disposable stuff → local disk**: the code checkout (`~/SATG`, trivially
  re-clonable) and the blobfuse **cache** (`/mnt/blobcache`, on the ephemeral temp
  disk).
- **Keep the node alive** during long runs by setting the cluster's **min nodes = 1**
  (Step 0) — otherwise it can scale to zero *under you* mid-run.
- Training is **`--resume`-safe** and checkpoints live on blob, so even a recycle
  only costs ≤2000 iters of the in-flight run.

> ⚠️ **Performance note.** The full working set (GTA5 ~57 GB + Cityscapes ~13 GB +
> heatmaps ~95 GB) is far larger than the local blobfuse cache (~45 GB on `/mnt`),
> so training reads that miss cache go over the network to blob. It **works**, but
> expect training to run slower than local-disk estimates. If it's too slow, either
> use a cluster SKU with a bigger temp disk (larger cache) or cut scope (smaller
> GTA5 subset / Phase-8 baselines only, which need no heatmaps).

---

## 0. Node facts + keep the node alive

| Field | Value |
|-------|-------|
| Public IP | `20.125.122.201` |
| SSH port | `50000` |
| SSH key | `~\.ssh\id_rsa_azureml` (the `chessmamba-azureml` keypair) |
| Login user | `azureuser` |
| Resource group | `pranav-rg` |
| Compute cluster | `chesstraining` |
| Region | `westus3` |

🖥️ **PowerShell (local)** — set these once *in your current window*:

```powershell
$SATG_HOST = "20.125.122.201"
$SATG_PORT = "50000"
$SATG_USER = "azureuser"
$SATG_KEY  = "$HOME\.ssh\id_rsa_azureml"
$RG        = "pranav-rg"
$LOC       = "westus3"
```

**Keep the cluster from scaling to zero mid-run.** In **Azure ML Studio** → *Compute*
→ *Compute clusters* → **chesstraining** → *Edit* → set **Minimum number of nodes = 1**
→ Save. (This bills continuously while set — drop it back to 0 when you're done.)

CLI alternative (needs the `ml` extension + your workspace name):

```powershell
az extension add -n ml
az ml compute update -g $RG -n chesstraining --workspace-name <your-workspace> --min-instances 1
```

---

## 1. Connect

🖥️ **PowerShell (local):**

```powershell
ssh -i $SATG_KEY -p $SATG_PORT "$($SATG_USER)@$($SATG_HOST)"
```

If it prompts for a **password**, `-i` is pointing at the wrong file — fix `$SATG_KEY`
(your key is `~\.ssh\id_rsa_azureml`).

---

## 2. Create the Blob container (persistent storage)

🖥️ **PowerShell (local)** — create a dedicated storage account + container
(install CLI first if needed: `winget install -e --id Microsoft.AzureCLI`, then
reopen PowerShell):

```powershell
az login
$SA = "satg" + (Get-Random -Maximum 999999)      # storage acct name: globally unique, lowercase
az storage account create -g $RG -n $SA -l $LOC --sku Standard_LRS --kind StorageV2
$KEY = az storage account keys list -g $RG -n $SA --query "[0].value" -o tsv
az storage container create --account-name $SA --account-key $KEY -n satg-data

# print the two values you'll paste on the node (keep them private):
Write-Host "STORAGE ACCOUNT: $SA"
Write-Host "STORAGE KEY:     $KEY"
```

Copy the **account name** and **key** — you'll paste them into the node config next.
(~200 GB on Standard_LRS blob ≈ a few $/month plus transactions.)

---

## 3. Mount the container on the node (blobfuse2)

☁️ **Bash (on the node)** — install blobfuse2 if missing, set up a cache on the
ephemeral temp disk, write the config, and mount:

```bash
# 3a. install blobfuse2 (skip if already present)
command -v blobfuse2 >/dev/null || {
  wget -q "https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb" -O /tmp/pmp.deb
  sudo dpkg -i /tmp/pmp.deb && sudo apt-get update && sudo apt-get install -y blobfuse2
}

# 3b. cache dir on the temp disk (safe to lose)
sudo mkdir -p /mnt/blobcache && sudo chown "$USER:$USER" /mnt/blobcache

# 3c. write the blobfuse2 config — EDIT the two REPLACE_ lines afterwards
cat > ~/blobfuse2.yaml <<'YAML'
components: [libfuse, file_cache, attr_cache, azstorage]
libfuse:
  attribute-expiration-sec: 120
  entry-expiration-sec: 120
file_cache:
  path: /mnt/blobcache
  timeout-sec: 600
  max-size-mb: 45000          # ~45 GB cache; keep under /mnt free space
attr_cache:
  timeout-sec: 120
azstorage:
  type: block
  mode: key
  account-name: REPLACE_ACCOUNT
  account-key: REPLACE_KEY
  endpoint: https://REPLACE_ACCOUNT.blob.core.windows.net
  container: satg-data
YAML

# 3d. paste your values in (replace both placeholders):
nano ~/blobfuse2.yaml         # set account-name (x2, incl. endpoint) and account-key

# 3e. mount
mkdir -p ~/blob
blobfuse2 mount ~/blob --config-file="$HOME/blobfuse2.yaml"
ls -la ~/blob                 # should succeed (empty container = empty listing)
```

> If mount fails, run it in the foreground to see the error:
> `blobfuse2 mount ~/blob --config-file=$HOME/blobfuse2.yaml --foreground`

---

## 4. Get the code + wire it to blob

☁️ **Bash (on the node)** — clone the repo locally (fast, re-clonable), then point
`data/`, `checkpoints/`, and `cloud/logs/` at the blob mount via symlinks so all
heavy + valuable output lands on persistent storage:

```bash
# code lives locally; it's cheap to re-clone if the node recycles
cd ~ && git clone <your-SATG-repo-url> SATG && cd ~/SATG

# persistent dirs on blob
mkdir -p ~/blob/data ~/blob/checkpoints ~/blob/logs

# symlink them into the repo (default config paths then "just work")
ln -sfn ~/blob/data        data
ln -sfn ~/blob/checkpoints checkpoints
ln -sfn ~/blob/logs        cloud/logs

ls -la data checkpoints cloud/logs   # each should show '-> /home/azureuser/blob/...'
```

> If a node recycle wipes `~/SATG`, just re-clone and re-run the 3 `ln -sfn` lines —
> your data and checkpoints are still on blob.

---

## 5. Kaggle credentials (for the two Cityscapes datasets)

Token from <https://www.kaggle.com/settings/account> → **Create New API Token**.

🖥️ **PowerShell (local)** — upload it:

```powershell
scp -i $SATG_KEY -P $SATG_PORT "$HOME\Downloads\kaggle.json" "$($SATG_USER)@$($SATG_HOST):~/kaggle.json"
```

☁️ **Bash (on the node):**

```bash
mkdir -p ~/.kaggle && mv ~/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
```

---

## 6. Environment setup

☁️ **Bash (on the node):**

```bash
cd ~/SATG
bash cloud/setup.sh          # GPU check + installs requirements.txt -> ">>> GPU READY <<<"
```

If PyTorch reports CUDA unavailable, reinstall the matching build (check `nvidia-smi`):

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```

---

## 7. Download data + precompute heatmaps (writes to blob)

Everything below writes through the symlinks onto blob, so it persists.
☁️ **Bash (on the node):**

```bash
cd ~/SATG
export TMPDIR=/mnt                       # use the temp disk for extraction scratch
tmux new -s satg-data
bash cloud/prepare_data.sh 2>&1 | tee cloud/logs/prepare_data.log
# detach: Ctrl-B then D    |    reattach: tmux attach -t satg-data
```

What it does (all resume-safe — re-running continues where it stopped):

1. **GTA5** — `cloud/download_gta5.sh`: 10 image + 10 label ZIPs (~57 GB) →
   `data/GTA5/{images,labels}` (24,966 each), then maps IDs → Cityscapes trainIds.
2. **Cityscapes (your 2 Kaggle datasets)** — `cloud/download_cityscapes.sh`, in this
   order, laid out exactly as the loaders expect:
   | Kaggle slug | → lands at (on blob) |
   |-------------|-----------|
   | `chrisviviers/cityscapes-leftimg8bit-trainvaltest` | `data/cityscapes/leftImg8bit/{train,val,test}/…` |
   | `kclaude/gtfine-trainvaltest` | `data/cityscapes/gtFine/{train,val}/…` |
   Then auto-generates `*_gtFine_labelTrainIds.png` from the label PNGs.
3. **Heatmaps** — 4 structural-prior components for all 2,975 train images (~95 GB
   of `.npy`, written next to the images on blob).
4. **Validation** — asserts 2,975 train images and 11,900 heatmaps.

Expected tail: `>>> CITYSCAPES READY <<<` … `>>> HEATMAP VALIDATION PASSED <<<`.

> Uploading ~165 GB to blob takes a while, and blobfuse write throughput varies.
> It's a one-time cost; subsequent runs read from blob (cached on `/mnt`).

---

## 8. WandB

☁️ **Bash (on the node):** `wandb login` (token at <https://wandb.ai/authorize>).
Offline instead? `wandb offline`, then `wandb sync cloud/logs` later.

---

## 9. Train (eviction/recycle-safe)

☁️ **Bash (on the node):**

```bash
cd ~/SATG
tmux new -s satg-train
bash cloud/run_phase8.sh      # baselines (6 runs)
bash cloud/run_phase9.sh      # SATG + ablations (~40 runs)
```

Detach: **Ctrl-B, D**. Reattach: `tmux attach -t satg-train`.

**If the node recycles** (or you re-attach after a drop): reconnect, remount blob if
needed, re-clone code + re-link if `~/SATG` was wiped, then re-run the **same**
command. Finished runs (`checkpoints/<run>/.done` on blob) skip; the interrupted run
resumes from its `last.pth`; the rest run fresh.

```bash
# after a recycle, the recovery is just:
blobfuse2 mount ~/blob --config-file=$HOME/blobfuse2.yaml    # if unmounted
cd ~/SATG || (cd ~ && git clone <repo> SATG && cd SATG && \
  ln -sfn ~/blob/data data && ln -sfn ~/blob/checkpoints checkpoints && ln -sfn ~/blob/logs cloud/logs)
bash cloud/run_phase9.sh
```

Single run / overrides:

```bash
python -m training.trainer --config configs/satg_hard.yaml --resume seed=42
python -m training.trainer --config configs/satg_hard.yaml \
    --run_name grid_tc0.95_ts0.70_seed42 \
    trust_gate.tau_conf=0.95 trust_gate.tau_struct=0.70 seed=42
```

### After training — figures + tables (Phase 10)

Once runs finish, generate the repo's analysis outputs (all write to blob via the
`visualizations/` dir if you symlink it, or local + download). ☁️ **on the node:**

```bash
# 1. Qualitative 1x5 trust-mask panels for the primary SATG variants
for v in satg_hard satg_soft_weight satg_soft_label; do
  python -m visualization.visualize \
      --checkpoint checkpoints/${v}_seed42/best.pth \
      --config configs/${v}.yaml \
      --num_images 10 --output_dir visualizations/
done

# 2. Trust-coverage-over-time plots from each run's metrics.csv
python - <<'PY'
import glob, os
from visualization.plot_metrics import plot_coverage_over_time
os.makedirs("visualizations/training_metrics", exist_ok=True)
for csv in glob.glob("checkpoints/*/metrics.csv"):
    run = os.path.basename(os.path.dirname(csv))
    out = f"visualizations/training_metrics/{run}_coverage.png"
    try:
        plot_coverage_over_time(csv, out); print("wrote", out)
    except Exception as e:
        print("skip", run, "-", e)
PY

# 3. Per-experiment best mIoU with mean±std across seeds (for EXPERIMENTS.md)
python - <<'PY'
import glob, os, csv, re, statistics as st
from collections import defaultdict
runs = defaultdict(list)
for f in glob.glob("checkpoints/*/metrics.csv"):
    run = os.path.basename(os.path.dirname(f))
    exp = re.sub(r"_seed\d+$", "", run)
    best = 0.0
    for row in csv.DictReader(open(f)):
        v = row.get("val_miou", "")
        if v not in ("", None): best = max(best, float(v))
    if best: runs[exp].append(best)
print(f"{'experiment':32} {'n':>2} {'mean_mIoU':>9} {'std':>6}")
for exp, xs in sorted(runs.items()):
    s = st.pstdev(xs) if len(xs) > 1 else 0.0
    print(f"{exp:32} {len(xs):>2} {sum(xs)/len(xs):9.2f} {s:6.2f}")
PY
```

`run_phase8.sh` / `run_phase9.sh` also print their own summary tables, and every run
is in WandB under `satg-uda`. The final `README.md` / `EXPERIMENTS.md` write-ups
(Phase 10 T049–T_FINAL) are manual authoring from these numbers.

Optional — validate the code with the unit tests (no GPU needed, fast):

```bash
pytest -q
```

---

## 10. Monitor

☁️ **Bash (on the node):**

```bash
tail -f cloud/logs/phase8_source_only_seed42.log
grep "★ New best" cloud/logs/*.log | tail
watch -n2 nvidia-smi
df -h /mnt/blobcache          # keep an eye on cache disk usage
```

---

## 11. Get results

Checkpoints + logs are already on **blob** (durable). Two ways to retrieve:

🖥️ **PowerShell (local)** — pull straight from blob (no node needed):

```powershell
az storage blob download-batch --account-name $SA --account-key $KEY `
    -s satg-data/checkpoints -d .\checkpoints
az storage blob download-batch --account-name $SA --account-key $KEY `
    -s satg-data/logs -d .\cloud\logs
```

Or scp from the node's blob mount (while it's up):

```powershell
scp -i $SATG_KEY -P $SATG_PORT -r "$($SATG_USER)@$($SATG_HOST):~/blob/checkpoints/*" .\checkpoints\
```

**When done:** set the cluster **min nodes back to 0** (Studio → Compute) so it stops
billing. Your data + checkpoints remain in the blob container.

---

## 12. Troubleshooting

| Symptom | Fix |
|--------|-----|
| **`export` / bash command fails in PowerShell** | You ran a ☁️ node command locally. Node commands go inside the SSH session; local env vars use `$NAME = "value"`. |
| **`az` not recognized** | `winget install -e --id Microsoft.AzureCLI`, then **reopen** PowerShell. |
| **SSH asks for a password** | `-i` points at a missing key — set `$SATG_KEY = "$HOME\.ssh\id_rsa_azureml"`. |
| **blobfuse mount fails** | Re-run with `--foreground` to see the error. Common causes: wrong account/key, container name ≠ `satg-data`, or `/mnt/blobcache` missing. |
| **Node vanished / SSH died mid-run** | Cluster scaled down. Set **min nodes = 1** (Step 0). Recover via the Step 9 recycle block; data/checkpoints are safe on blob. |
| **Training very slow / high disk wait** | Working set > cache. Raise `file_cache.max-size-mb` if `/mnt` allows, use a bigger-temp-disk SKU, or cut scope (GTA5 subset / baselines only). |
| **`No space left` during download** | Extraction scratch filled `/tmp`. `export TMPDIR=/mnt` before `prepare_data.sh`; keep `/mnt/blobcache` under the temp-disk size. |
| **Cityscapes counts < 2975** | Kaggle mirror shipped a subset. `ls ~/blob/data/cityscapes/leftImg8bit/train`. |
| **CUDA OOM** | Append `training.batch_size=2 training.crop_size="[384,384]"` to the run command. |

---

## Appendix — quickstart

🖥️ **PowerShell (local):**

```powershell
$SATG_HOST="20.125.122.201"; $SATG_PORT="50000"; $SATG_USER="azureuser"
$SATG_KEY="$HOME\.ssh\id_rsa_azureml"; $RG="pranav-rg"; $LOC="westus3"
# (Studio: set chesstraining min nodes = 1)
az login
$SA="satg"+(Get-Random -Maximum 999999)
az storage account create -g $RG -n $SA -l $LOC --sku Standard_LRS --kind StorageV2
$KEY=az storage account keys list -g $RG -n $SA --query "[0].value" -o tsv
az storage container create --account-name $SA --account-key $KEY -n satg-data
Write-Host "ACCOUNT: $SA`nKEY: $KEY"
ssh -i $SATG_KEY -p $SATG_PORT "$($SATG_USER)@$($SATG_HOST)"
```

☁️ **Bash (on the node), after SSH:**

```bash
# mount blob (after editing ~/blobfuse2.yaml with your account+key — see Step 3)
sudo mkdir -p /mnt/blobcache && sudo chown "$USER:$USER" /mnt/blobcache
mkdir -p ~/blob && blobfuse2 mount ~/blob --config-file=$HOME/blobfuse2.yaml
mkdir -p ~/blob/data ~/blob/checkpoints ~/blob/logs
cd ~ && git clone <repo> SATG && cd ~/SATG
ln -sfn ~/blob/data data && ln -sfn ~/blob/checkpoints checkpoints && ln -sfn ~/blob/logs cloud/logs
mkdir -p ~/.kaggle && chmod 600 ~/.kaggle/kaggle.json      # after scp'ing token
bash cloud/setup.sh
export TMPDIR=/mnt
tmux new -s satg
bash cloud/prepare_data.sh
wandb login
bash cloud/run_phase8.sh
bash cloud/run_phase9.sh
```

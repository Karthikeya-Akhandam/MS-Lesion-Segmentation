# MS Lesion Segmentation — Step-by-Step Usage Guide

This guide covers everything needed to run the project from scratch, whether you are on
Google Colab or a local/university machine.

---

## What the three notebooks do

| Notebook | Run it when… |
|---|---|
| `Data_Downloader.ipynb` | You need to download the datasets for the first time |
| `Project_Training.ipynb` | You want to preprocess data and train the model |
| `Project_Demo.ipynb` | You want to run the interactive clinical demo |

Run them **in the order above**. Training depends on downloaded data; the demo depends on a trained model.

---

## Option A — Google Colab (first time)

### Step 1 — Download the datasets

1. Open `Data_Downloader.ipynb` in Colab (`File → Upload notebook` or open from Drive).
2. In **Cell 1**, change the line:
   ```python
   ENVIRONMENT = "local"
   ```
   to:
   ```python
   ENVIRONMENT = "colab"
   ```
3. Run **Cell 1** — it mounts your Google Drive and sets the save folder to
   `My Drive/brain_dataset/`.
4. Run **Cell 2** — downloads all datasets directly to your Drive using `aria2c`
   and the Python `requests` library. This takes 30–90 minutes depending on
   connection speed. The cell is safe to re-run — already-downloaded files are
   skipped automatically.
5. Skip **Cell 3** (that is for local mode only).

### Step 2 — Train the model

1. Open `Project_Training.ipynb` in Colab.
2. In **Cell 2**, confirm (or set):
   ```python
   ENVIRONMENT = "colab"
   ```
3. Run all cells in order from top to bottom (`Runtime → Run all`).
   - Cell 2: mounts Drive, sets paths.
   - Cell 4: installs dependencies (`monai`, `SimpleITK`, `nibabel`, `tqdm`).
   - Cell 6: extracts ZIPs from `My Drive/brain_dataset/` to fast local Colab storage.
   - Cells 8–14: discovers data pairs, runs N4 bias correction + skull stripping.
   - Cells 16–19: defines the model, loss, and ablation config.
   - Cell 21: defines the training engine (does NOT start training yet).
4. To start training, scroll to **Cell 21** and uncomment the last line:
   ```python
   train_model(processed_train_files)
   ```
   Then run that cell. Training runs for 100 epochs and saves:
   - `My Drive/MS_Project/models/best_model.pth` — best checkpoint by validation Dice
   - `My Drive/MS_Project/models/checkpoint_latest.pth` — latest epoch (for resuming)

   > **Colab tip:** Enable a T4 GPU via `Runtime → Change runtime type → T4 GPU`
   > before running. Training on CPU will be 20–30× slower.

### Step 3 — Run the demo

1. Open `Project_Demo.ipynb` in Colab.
2. Make sure `best_model.pth` is in `My Drive/MS_Project/models/` (produced by Step 2).
3. Run all cells. A Gradio link will appear — click it to open the dashboard.

---

## Option B — Local / University Machine (first time)

### Prerequisites

Install dependencies once:

```bash
pip install monai[all] SimpleITK nibabel tqdm gradio matplotlib torch gdown
```

If you have a CUDA GPU, install the matching PyTorch version from pytorch.org first.

### Step 1 — Download the datasets from your Google Drive

> You need to have already downloaded the dataset ZIPs to your Google Drive
> (either by running Data_Downloader in Colab previously, or by having uploaded
> them manually).

1. Open `Data_Downloader.ipynb` in JupyterLab or VS Code.
2. In **Cell 1**, confirm:
   ```python
   ENVIRONMENT = "local"
   ```
3. Run **Cell 1** — creates `./data/raw/` in the current folder.
4. Skip **Cell 2** (Colab-only).
5. Open **Cell 3**. Fill in your Google Drive share links:
   - Go to Google Drive in your browser.
   - Right-click each ZIP file → **Get link** → **Anyone with the link** → Copy.
   - Paste each full URL (the long `https://drive.google.com/file/d/…` link) into
     the `GDRIVE_SHARE_LINKS` dictionary, replacing each `PASTE_ID_HERE`.
   - Example of a filled entry:
     ```python
     ("https://drive.google.com/file/d/1aBcD2eFgHiJkLm3NoPqRs/view?usp=sharing", "MSLesSeg_Part1.zip"),
     ```
6. Run **Cell 3** — `gdown` downloads each ZIP and extracts it automatically.
   Files already downloaded are skipped on re-run.

### Step 2 — Train the model

1. Open `Project_Training.ipynb`.
2. In **Cell 2**, confirm:
   ```python
   ENVIRONMENT = "local"
   ```
3. Run all cells in order. Training saves model checkpoints to `./models/` in the
   current directory:
   - `./models/best_model.pth`
   - `./models/checkpoint_latest.pth`

### Step 3 — Run the demo

1. Open `Project_Demo.ipynb`.
2. Update the model path in the demo notebook to point to `./models/best_model.pth`.
3. Run all cells.

---

## Saving the locally trained model to Google Drive

After training locally you will want to back up the model weights to Drive.
There are two ways:

### Method 1 — rclone (automatic, runs from the notebook)

This is already wired into `Project_Training.ipynb` — the last code cell calls
`sync_models_to_drive()` automatically when `ENVIRONMENT = "local"`.

**One-time setup (do this on your home machine):**

```bash
# 1. Install rclone
#    Windows: download the .zip from https://rclone.org/downloads/, extract rclone.exe
#    Linux:   curl https://rclone.org/install.sh | sudo bash

# 2. Configure (opens a browser for Google sign-in)
rclone config
# When prompted:
#   → n  (New remote)
#   → Name: gdrive
#   → Type: drive   (Google Drive)
#   → Follow the prompts, sign in with your Google account
#   → Accept defaults for everything else

# 3. Test it works
rclone ls gdrive:
```

**Copy the config to the university machine (no browser needed there):**

| OS | Config file location |
|---|---|
| Windows | `C:\Users\<you>\AppData\Roaming\rclone\rclone.conf` |
| Linux / Mac | `~/.config/rclone/rclone.conf` |

Copy this file to the same path on the university machine. rclone will authenticate
using the stored tokens — no browser or Google sign-in required on that machine.

After setup, every time you run `Project_Training.ipynb` locally the last cell will
automatically sync `./models/` to `My Drive/MS_Project/models/`.

---

### Method 2 — Manual upload (no setup needed)

If you cannot install rclone on the university machine, after training finishes:

1. Locate `./models/best_model.pth` (and optionally `checkpoint_latest.pth`).
2. Upload them to Google Drive manually via the browser, or copy them to a USB drive.
3. Place them in `My Drive/MS_Project/models/` so `Project_Demo.ipynb` in Colab
   can find them at the expected path.

---

## Switching between FLAIR-only and multi-modal training

By default the model trains on FLAIR only (`ABLATION = "A"`). To use multiple MRI modalities:

1. In **Cell 19** of `Project_Training.ipynb`, change:
   ```python
   ABLATION = "A"   # FLAIR only
   ```
   to:
   ```python
   ABLATION = "B"   # FLAIR + T1
   # or
   ABLATION = "C"   # FLAIR + T1 + T2
   ```
2. Re-run Cell 19 onwards. Pairs where the required modality files are missing
   are silently dropped (a count is printed).

> **Note:** If the dataset folders do not contain T1/T2 files alongside FLAIR,
> ablations B and C will find 0 pairs and fall back to 0 training samples.
> In that case, stick with `ABLATION = "A"`.

---

## Common issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: monai` | Run `pip install monai[all] SimpleITK nibabel tqdm` |
| `gdown` download stuck / 0 bytes | The share link is not set to "Anyone with the link" — re-check permissions |
| CUDA out of memory | Lower `BATCH_SIZE` to 1 or `GPU_BATCH` to 3 in Cell 14 |
| `Total Aligned Pairs: 0` | Data folder structure doesn't match expected naming — run Cell 8 (directory mapper) to inspect what's there |
| rclone: `Failed to create file system` | Remote name mismatch — make sure you named it `gdrive` during `rclone config` |

# 🔬 DermFed — Federated Skin Cancer Detection System

> **Privacy-preserving AI for dermatology.**  
> Multiple simulated hospitals collaboratively train a MobileNetV2 model on the HAM10000 dataset using Federated Learning (Flower + PyTorch) — *without ever sharing raw patient images*.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Dataset Setup (HAM10000)](#3-dataset-setup-ham10000)
4. [Environment Setup (Windows / VS Code)](#4-environment-setup-windows--vs-code)
5. [Data Partitioning](#5-data-partitioning)
6. [Running the System](#6-running-the-system)
7. [Streamlit Dashboard](#7-streamlit-dashboard)
8. [Project Structure](#8-project-structure)
9. [Configuration Reference](#9-configuration-reference)
10. [Troubleshooting](#10-troubleshooting)
11. [Extending the Project](#11-extending-the-project)

---

## 1. Project Overview

| Component | Technology |
|---|---|
| FL Framework | [Flower (flwr)](https://flower.dev/) |
| ML Backbone  | PyTorch · MobileNetV2 (ImageNet pre-trained) |
| Dataset      | [HAM10000](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) — 10,015 dermoscopic images · 7 classes |
| Dashboard    | Streamlit + Plotly |
| Platform     | Windows 10 / 11 · Python 3.10 or 3.11 |

### Federated Learning Flow

```
┌──────────────┐    weights     ┌─────────────────────┐
│  Hospital 0  │◄──────────────►│                     │
├──────────────┤                │   FL Server         │
│  Hospital 1  │◄──────────────►│   (FedAvg)          │
├──────────────┤                │   Aggregates &      │
│  Hospital 2  │◄──────────────►│   broadcasts        │
└──────────────┘  (NO raw data) └─────────────────────┘
       │
       └──► Streamlit Dashboard (real-time monitoring + inference)
```

Each hospital:
1. Receives the current **global model weights** from the server.
2. Trains locally on its **private patient data**.
3. Sends back only the **model weight updates** (gradients stay local).

---

## 2. Architecture

```
DermFed/
├── partition_data.py   # Split HAM10000 into hospital silos
├── utils.py            # Dataset class, model factory, train/eval helpers
├── server.py           # Flower FL server (FedAvg + metric logging)
├── client.py           # Flower FL client (one per hospital)
├── app.py              # Streamlit dashboard (Simulation + Inference tabs)
├── requirements.txt
├── launch_all.bat      # Windows one-click launcher
│
├── data/
│   ├── raw/            # ← Place HAM10000 files HERE (see Section 3)
│   └── partitions/     # Auto-created by partition_data.py
│
├── models/
│   └── global_model.pt # Auto-saved after each FL round
│
└── results/
    └── fl_metrics.csv  # Per-round accuracy/loss (read by dashboard)
```

---

## 3. Dataset Setup (HAM10000)

> ⚠️ **The dataset is NOT included.** You must download it from Kaggle.

### Step 1 — Create a Kaggle Account

Go to [https://www.kaggle.com](https://www.kaggle.com) and sign up (free).

### Step 2 — Download the Dataset

1. Visit: [https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)
2. Click **Download** (top-right). This downloads a ZIP file (~2 GB).

> **Alternative — Kaggle CLI** (faster for large files):
> ```bat
> pip install kaggle
> :: Place your kaggle.json in C:\Users\<YourName>\.kaggle\
> kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection
> ```

### Step 3 — Extract and Place Files

After downloading and unzipping, you will have these files:

```
HAM10000_metadata.csv
HAM10000_images_part_1/    (folder with ~5000 .jpg images)
HAM10000_images_part_2/    (folder with ~5000 .jpg images)
```

**Copy them into the project so the layout looks exactly like this:**

```
DermFed/
└── data/
    └── raw/
        ├── HAM10000_metadata.csv
        ├── HAM10000_images_part_1/
        │   ├── ISIC_0024306.jpg
        │   ├── ISIC_0024307.jpg
        │   └── ...
        └── HAM10000_images_part_2/
            ├── ISIC_0029306.jpg
            └── ...
```

> 💡 **Create the `data/raw/` folder** if it doesn't exist yet:
> ```bat
> mkdir data\raw
> ```

### HAM10000 — The 7 Classes

| Label | Code | Description | Risk |
|---|---|---|---|
| 0 | akiec | Actinic Keratoses | Pre-cancerous |
| 1 | bcc   | Basal Cell Carcinoma | Malignant |
| 2 | bkl   | Benign Keratosis-like | Benign |
| 3 | df    | Dermatofibroma | Benign |
| 4 | mel   | Melanoma | **High-risk malignant** |
| 5 | nv    | Melanocytic Nevi | Benign (common mole) |
| 6 | vasc  | Vascular Lesions | Benign |

---

## 4. Environment Setup (Windows / VS Code)

### Prerequisites

| Requirement | Download |
|---|---|
| Python 3.10 or 3.11 | [python.org](https://www.python.org/downloads/) |
| Git (optional) | [git-scm.com](https://git-scm.com/) |
| VS Code | [code.visualstudio.com](https://code.visualstudio.com/) |
| VS Code Python Extension | Install from VS Code Extensions tab |

> ⚠️ During Python installation on Windows, **tick "Add Python to PATH"**.

### Step 1 — Open the Project in VS Code

```bat
:: Open terminal in VS Code (Ctrl + `)
cd C:\path\to\DermFed
```

### Step 2 — Create a Virtual Environment

```bat
python -m venv venv
```

### Step 3 — Activate the Virtual Environment

```bat
venv\Scripts\activate
```

Your terminal prompt should now show `(venv)`.

> In VS Code, press `Ctrl+Shift+P` → **"Python: Select Interpreter"** → choose `.\venv\Scripts\python.exe`

### Step 4 — Install Dependencies

```bat
pip install --upgrade pip
pip install -r requirements.txt
```

This installs: `flwr`, `torch`, `torchvision`, `streamlit`, `plotly`, `scikit-learn`, `pandas`, `Pillow`.

> ⏱️ First install takes ~5–10 minutes (PyTorch is large).

#### GPU Support (Optional)

If you have an NVIDIA GPU with CUDA 12.1:

```bat
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Step 5 — Create Required Directories

```bat
mkdir data\raw
mkdir data\partitions
mkdir models
mkdir results
```

---

## 5. Data Partitioning

After placing the HAM10000 files in `data/raw/`, run:

```bat
python partition_data.py --n_clients 3 --strategy iid
```

| Argument | Options | Description |
|---|---|---|
| `--n_clients` | 2–5 | Number of hospital silos to create |
| `--strategy` | `iid` or `non_iid` | Data distribution type |

**IID** (Independent & Identically Distributed): Each hospital gets a balanced random sample — simulates ideal conditions.

**Non-IID**: Each hospital is biased toward certain lesion types (Dirichlet distribution) — simulates real-world specialisation.

**Expected output:**

```
Hospital 0  (2,800 samples)
   akiec   120  ██████
   bcc     180  █████████
   bkl     430  █████████████████████
   ...
  ✓  2,800 images  →  data\partitions\hospital_0

Hospital 1  ...
Hospital 2  ...
Global test set: 1,502 samples
```

---

## 6. Running the System

> You need **4 separate terminal windows** running simultaneously.  
> In VS Code: click the `+` button in the terminal panel to open new terminals.

### Option A — Manual (Recommended for learning)

**Terminal 1 — FL Server**
```bat
venv\Scripts\activate
python server.py --rounds 10 --n_clients 3
```
> Wait until you see: `INFO flwr 2024-... Flower ECE: gRPC server running`

**Terminal 2 — Hospital 0**
```bat
venv\Scripts\activate
python client.py --hospital_id 0
```

**Terminal 3 — Hospital 1**
```bat
venv\Scripts\activate
python client.py --hospital_id 1
```

**Terminal 4 — Hospital 2**
```bat
venv\Scripts\activate
python client.py --hospital_id 2
```

**Terminal 5 — Streamlit Dashboard**
```bat
venv\Scripts\activate
streamlit run app.py
```

The dashboard opens automatically at **http://localhost:8501**

### Option B — One-click Launcher (Windows)

```bat
.\launch_all.bat 3 10
```
Arguments: `[n_clients]` `[n_rounds]`  
This opens all 5 windows automatically with a 3–4 second stagger.

### Server Options

```bat
python server.py --rounds 20 --n_clients 3 --address 0.0.0.0:8080
```

### Client Options

```bat
python client.py --hospital_id 0 --server 127.0.0.1:8080
```

---

## 7. Streamlit Dashboard

Navigate to **http://localhost:8501** after running `streamlit run app.py`.

### Tab A — Simulation Monitor

- 🏥 Visual hospital grid showing all active clients
- 📊 Real-time line chart of **Global Accuracy** and **Val Loss** per FL round
- 🔢 Metric cards: rounds complete, current accuracy, loss, active clients
- 🔄 Auto-refresh toggle (5 / 10 / 30 / 60 second intervals)
- 📋 Quick-start command reference

### Tab B — Image Inference

1. Upload any dermatoscopic `.jpg` / `.png` image
2. The global model outputs:
   - **Predicted diagnosis** with confidence %
   - **Full probability bar chart** across all 7 classes
3. Uses `models/global_model.pt` (auto-saved after each FL round)

> 💡 The inference tab works with randomly-initialised weights before training, but predictions won't be meaningful until the model has trained for several rounds.

---

## 8. Project Structure

```
DermFed/
│
├── 📄 app.py                # Streamlit dashboard (Simulation + Inference)
├── 📄 server.py             # FL Server — FedAvg aggregation + metric logging
├── 📄 client.py             # FL Client — local training at each hospital
├── 📄 utils.py              # Dataset, model, train/eval helpers (shared)
├── 📄 partition_data.py     # HAM10000 → hospital silos
│
├── 📄 requirements.txt      # Python dependencies
├── 📄 launch_all.bat        # Windows one-click launcher
├── 📄 README.md
│
├── 📁 data/
│   ├── 📁 raw/              # ← Place HAM10000 files here
│   │   ├── HAM10000_metadata.csv
│   │   ├── HAM10000_images_part_1/
│   │   └── HAM10000_images_part_2/
│   └── 📁 partitions/       # Auto-created
│       ├── hospital_0/
│       │   ├── images/
│       │   └── metadata.csv
│       ├── hospital_1/
│       ├── hospital_2/
│       └── global_test/
│
├── 📁 models/
│   └── global_model.pt      # Auto-saved after each round
│
└── 📁 results/
    └── fl_metrics.csv       # Per-round metrics (read by dashboard)
```

---

## 9. Configuration Reference

### `utils.py`

| Constant | Default | Description |
|---|---|---|
| `NUM_CLASSES` | 7 | HAM10000 has 7 lesion types |
| `IMG_SIZE` | 224 | Input resolution for MobileNetV2 |
| `BATCH_SIZE` | 32 | Reduce to 16 if you get memory errors |

### `client.py`

| Constant | Default | Description |
|---|---|---|
| `LOCAL_EPOCHS` | 2 | Epochs of local training per round |
| `LEARNING_RATE` | 1e-3 | Adam learning rate |

### `server.py`

| Constant | Default | Description |
|---|---|---|
| `fraction_fit` | 1.0 | Fraction of clients used per round |
| `fraction_evaluate` | 1.0 | Fraction evaluated per round |

### `partition_data.py`

| Constant | Default | Description |
|---|---|---|
| `DATA_DIR` | `data/raw` | Source of HAM10000 files |
| `OUTPUT_DIR` | `data/partitions` | Destination for silos |

---

## 10. Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'flwr'`
```bat
venv\Scripts\activate          # Make sure venv is active
pip install -r requirements.txt
```

### ❌ `FileNotFoundError: Metadata CSV not found`
Make sure `data/raw/HAM10000_metadata.csv` exists. See [Section 3](#3-dataset-setup-ham10000).

### ❌ `Address already in use` on port 8080
Another process is using port 8080. Either kill it or use a different port:
```bat
python server.py  --address 0.0.0.0:8081
python client.py  --server 127.0.0.1:8081
```

### ❌ `RuntimeError: CUDA out of memory`
Edit `utils.py` and reduce `BATCH_SIZE` from `32` to `16` or `8`.

### ❌ Clients connect but training never starts
The server waits until **all** `min_fit_clients` connect. Make sure you start exactly `N` client processes where `N == --n_clients` passed to the server.

### ❌ `OSError: [WinError 10048]` on port 8080
A previous server process is still running. Open Task Manager, find `python.exe`, and end it. Or restart your computer.

### ❌ Streamlit shows blank chart
The `results/fl_metrics.csv` is empty or has no `val_acc` column yet. The chart appears after the **first complete FL round** (both fit + evaluate phases finish).

### 🐢 Training is very slow on CPU
- MobileNetV2 on CPU trains at ~1–3 min/epoch on typical data subsets.
- Reduce `LOCAL_EPOCHS` in `client.py` to `1`.
- Reduce `BATCH_SIZE` to `16`.
- Use fewer rounds: `--rounds 5`.

---

## 11. Extending the Project

### Add More Clients
```bat
python partition_data.py --n_clients 5 --strategy non_iid
python server.py --n_clients 5 --rounds 15
:: Then start 5 client terminals: hospital_id 0..4
```

### Try Non-IID Distribution
```bat
python partition_data.py --n_clients 3 --strategy non_iid
```

### Swap the Backbone
In `utils.py`, replace `models.mobilenet_v2(...)` with any `torchvision.models` architecture (e.g., EfficientNet-B0, ResNet-50).

### Add Differential Privacy
Flower supports DP natively:
```python
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping
```

### Evaluate on Global Test Set
```python
# In utils.py — evaluate on data/partitions/global_test/
train_loader, val_loader = get_loaders("data/partitions/global_test")
```

---

## License

This project is for **educational and research purposes only**.  
The HAM10000 dataset is © its respective authors; see the [Kaggle dataset page](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) for licensing.

> ⚕️ **Medical Disclaimer** — DermFed is a research prototype. It is NOT a medical device and must NOT be used for clinical diagnosis or treatment decisions.

---

*Built with 🔬 Flower · PyTorch · Streamlit · HAM10000*

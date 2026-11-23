# FracPIC-GCN

**FracPIC-GCN: Fractional Projection Influence Contradiction Graph Convolutional Network**

This repository provides a PyTorch implementation of FracPIC-GCN for robust semi-supervised node classification under scarce and noisy labels.

## 1. Overview

In semi-supervised node classification on graphs, incorrect training labels can interfere with model parameter updates and cause shifts in hidden node representations. Such shifts may change the relationships among nodes in the representation space and, in turn, affect label-noise detection and correction based on those relationships. Consequently, identifying and correcting mislabeled nodes only at the supervision stage may not fully eliminate the influence of label noise on representation learning.

To address this issue, we first establish a **local perturbation-propagation analysis for fractional feature projection**. Under explicitly stated assumptions, we analyze the relationship between fractional-order weight gradients and standard first-order gradients, derive norm bounds for the fractional correction and projection-weight gradients, and investigate gradient perturbations induced by label flips and the resulting changes in node representations after one parameter update under a fixed training state and identical Adam histories.

Building on this analysis, we propose **FracPIC-GCN (Fractional Projection Influence Contradiction Graph Convolutional Network)**. Its core component, the **Fractional Graph Projection Layer (FGPL)**, introduces fractional-order parameter differentiation only into the **backward gradients of the first-layer feature-projection weights** in the main GCN classifier. Frobenius-norm clipping is used to control the magnitude of the fractional correction. FGPL preserves standard GCN forward feature projection and neighborhood aggregation without replacing the Adam optimizer. When the fractional order is set to 1, the standard first-order feature-projection pathway is recovered.

The model further combines auxiliary graph-structure estimation, structural and attribute-based dual-view influence contradiction, Gaussian mixture model (GMM)-based label-confidence estimation, soft-label refinement, and pseudo-label learning, allowing representation learning and noisy-supervision handling to operate within a unified training framework.

## 2. Installation

### Hardware and Software Environment

The experiments were conducted using an **NVIDIA GeForce RTX 4090D GPU**. The provided code is primarily intended for Linux x86-64 systems equipped with an NVIDIA GPU. The CUDA runtime and driver should be compatible with the installed PyTorch build.

| Component         | Version / Configuration    |
| ----------------- | -------------------------- |
| GPU               | NVIDIA GeForce RTX 4090D   |
| Operating system  | Linux x86-64 (recommended) |
| Python            | 3.10.20                    |
| PyTorch           | 2.5.1 (CUDA 12.4 build)    |
| PyTorch Geometric | 2.6.1                      |
| NumPy             | 1.26.4                     |
| SciPy             | 1.13.1                     |
| scikit-learn      | 1.5.2                      |

For the complete dependency list and version specifications, see [`environment.yml`](environment.yml) in the repository root.

### Clone the Repository and Install Dependencies

```bash
git clone https://github.com/WuJun5201314/FracPIC-GCN.git
cd FracPIC-GCN

conda env create -f environment.yml
conda activate fracpic-gcn

python -m pip check
python train.py --help
```

Run these commands from the repository root.

## 3. Datasets

The main experiments cover six graph node-classification datasets under two types of label noise: **Uniform** and **Pair**. The training-label noise rate is **20%**, and label corruption is applied only to training labels.

| Dataset      | `--dataset` | Training-label Ratio |
| ------------ | ----------- | -------------------: |
| Coauthor CS  | `cs`        |                   1% |
| Amazon Photo | `photo`     |                   1% |
| Cora         | `cora`      |                   5% |
| Pubmed       | `pubmed`    |                   1% |
| DBLP         | `dblp`      |                   1% |
| Citeseer     | `citeseer`  |                   5% |

### Option 1: Automatic Dataset Download

If a required dataset is not available locally, the dataset loader attempts to download it during training:

* **Cora and Citeseer:** `.npz` files from the [gnn-meta-attack data directory](https://github.com/danielzuegner/gnn-meta-attack/tree/master/data).
* **Pubmed:** `ind.pubmed.*` files from the [original GCN repository](https://github.com/tkipf/gcn/tree/master/gcn/data).
* **Coauthor CS, Amazon Photo, and DBLP:** downloaded through the [PyTorch Geometric dataset loaders](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html).

By default, the datasets are stored in the `data/` directory under the repository root. This option requires access to the original data sources on the first run. If the corresponding Personalized PageRank (PPR) cache is absent, the program also attempts to compute and save it. Because some datasets involve dense-matrix operations, generating PPR caches may require substantial memory and computation.

### Option 2: Download from Baidu Netdisk

For convenience, the complete shared `data` folder, including the six main datasets and precomputed PPR caches, is also available via Baidu Netdisk:

**[Download the FracPIC-GCN datasets from Baidu Netdisk](https://pan.baidu.com/s/1CUVdQknzSQzWfnY-O0ntJA?pwd=japr)**

**Extraction code:** `japr`

After downloading, place the **entire `data` folder** directly in the repository root, as shown below:

```text
FracPIC-GCN/
├── data/
│   ├── cora.npz
│   ├── citeseer.npz
│   ├── ind.pubmed.*
│   ├── cs/CS/
│   ├── photo/Photo/
│   ├── dblp/
│   └── *_ppr.pt              # Optional PPR caches
├── models/
├── dataset.py
├── train.py
└── run_fracpic_gcn.py
```

Avoid creating an extra nested directory such as `FracPIC-GCN/data/data/`.

## 4. Running Experiments

Run all commands below from the **`FracPIC-GCN/` repository root**.

### 4.1 Pre-run Check

```bash
python run_fracpic_gcn.py --gpu 0 --dry-run
```

This command checks the CUDA environment and displays the 12 predefined experiment configurations without launching full training.

### 4.2 Run a Single Experiment

The following example runs **Cora with Uniform label noise (20%)** on GPU 0, starting from random seed 11, for **one training and evaluation run**:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset cora \
  --noise uniform \
  --ptb_rate 0.2 \
  --label_rate 0.05 \
  --K 50 \
  --local_conflict_weight 0.9 \
  --warmup_epochs 30 \
  --scale1 1.0 \
  --temp 0.01 \
  --alpha 0.72 \
  --c 0.75 \
  --seed 11 \
  --runs 1
```

These parameters correspond to the Cora/Uniform configuration in `run_fracpic_gcn.py`, except that this example performs only one run. For other datasets or noise settings, refer to the `EXPERIMENTS` configurations in that script.

### 4.3 Main Training Arguments

| Argument           | Description                                                          |
| ------------------ | -------------------------------------------------------------------- |
| `--dataset`        | Dataset name                                                         |
| `--noise`          | `uniform` or `pair`                                                  |
| `--ptb_rate`       | Training-label noise probability                                     |
| `--label_rate`     | Training-label ratio                                                 |
| `--alpha`          | Fractional order, `0 < alpha <= 1`; `1.0` disables FGPL              |
| `--c`              | Frobenius-norm clipping threshold for the FGPL fractional correction |
| `--seed`, `--runs` | Initial training seed and number of consecutive-seed runs            |
| `--K`              | Top-K for retrieving candidate node relationships                    |
| `--warmup_epochs`  | Number of warm-up epochs for label-confidence estimation             |
| `--temp`           | Sharpening temperature for neighborhood predictions                  |

For the complete argument list, run `python train.py --help`.

### 4.4 Run All 12 Main Experiments

```bash
python run_fracpic_gcn.py --gpu 0 --max-retries 1
```

The experiment runner automatically executes **six datasets × two noise types**, with **five random seeds (11–15)** per configuration, and summarizes test accuracies. `--gpu 0` selects the first physical GPU; change the index to use a different GPU.

### 4.5 Experiment Outputs and Resuming

By default, the automated experiment runner writes its outputs to `fracpic_gcn_results/`:

| File          | Contents                                                                                 |
| ------------- | ---------------------------------------------------------------------------------------- |
| `results.csv` | Mean test accuracy, standard deviation, and results for five random seeds per experiment |
| `results.txt` | Text summary of the results                                                              |
| `runner.log`  | Environment and experiment execution logs                                                |
| `state.json`  | Completed-experiment state used for resuming interrupted runs                            |

Detailed training logs are also written to `logs/fracpic_gcn/<dataset>/`. Accuracies in the logs and CSV files are represented as fractions; for example, `0.881` corresponds to `88.1%`.

If the full experiment suite is interrupted, rerun the same command with the same output directory to skip completed configurations. To start a new, independent experiment suite, specify a different output directory:

```bash
python run_fracpic_gcn.py --gpu 0 --max-retries 1 \
  --output-dir fracpic_gcn_results_new
```

## 5. Results

The following table reports the **12 main FracPIC-GCN results from Table 2 of the manuscript**. The evaluation metric is **test node-classification accuracy (%, mean ± standard deviation over five runs)**. All settings use a 20% training-label noise rate.

| Dataset      | Noise Type | FracPIC-GCN Test Accuracy (%) |
| ------------ | ---------- | ----------------------------: |
| Coauthor CS  | Uniform    |                    88.1 ± 0.5 |
| Coauthor CS  | Pair       |                    86.9 ± 1.1 |
| Amazon Photo | Uniform    |                    88.3 ± 0.7 |
| Amazon Photo | Pair       |                    87.0 ± 0.6 |
| Cora         | Uniform    |                    80.4 ± 1.9 |
| Cora         | Pair       |                    79.7 ± 0.8 |
| Pubmed       | Uniform    |                    80.6 ± 0.5 |
| Pubmed       | Pair       |                    79.8 ± 0.5 |
| DBLP         | Uniform    |                    78.9 ± 1.4 |
| DBLP         | Pair       |                    80.3 ± 0.4 |
| Citeseer     | Uniform    |                    72.2 ± 0.4 |
| Citeseer     | Pair       |                    72.4 ± 1.1 |

These results are taken from Table 2 of the manuscript and are reported as the mean and standard deviation of test accuracy over five independent runs. This repository provides the corresponding experimental code and predefined configurations. Reproduced results may vary due to differences in hardware, software versions, and randomness.

## 6. Repository Structure

```text
FracPIC-GCN/
├── models/
│   ├── FGPL.py                 # Fractional graph projection layer
│   ├── GCN.py                  # Two-layer GCN classifier
│   └── FracPIC_GCN.py          # Main model and auxiliary graph-structure estimator
├── dataset.py                  # Dataset loading and preprocessing
├── train.py                    # Training, validation, and testing
├── run_fracpic_gcn.py          # Experiment runner for the 12 main settings
├── run.sh                      # Additional script included in the project
├── utils.py                    # Utility functions
├── environment.yml             # Conda environment configuration
├── .gitignore
└── README.md
```

The following directories may be used or generated during execution:

```text
data/                   # Datasets and PPR caches
logs/                   # Training logs
fracpic_gcn_results/    # Main experiment results
```

These data and output directories are not included in the current source-code package. They should be downloaded separately or generated during execution.


## 7. Acknowledgments

We thank the authors of [ICGNN](https://github.com/wayc04/ICGNN) for making their research code publicly available, which provided an important reference for this implementation. FracPIC-GCN extends this work by introducing fractional feature projection.

# FracPIC-GCN

FracPIC-GCN (Fractional Projection Influence Contradiction Graph Convolutional
Network) is a method for semi-supervised node classification with noisy labels.
Its fractional graph projection layer (FGPL) applies a fractional weight
gradient to the first feature projection of the main GCN classifier. The
forward pass uses standard GCN propagation.

This package covers six datasets under uniform and pair label noise, for 12
experimental settings. It includes entry points for FracPIC-GCN and the ICGNN
baseline. OGBN-Arxiv and Cornell are outside the scope of this package.

| Source file | Paper component |
| --- | --- |
| `models/FracPIC_GCN.py` | `FracPICGCN` (Section 4) and the `EstimateAdj` auxiliary graph estimator (Section 4.2) |
| `models/FGPL.py` | `FGPL`, the fractional graph projection layer (Section 4.3) |
| `models/GCN.py` | Two-layer GCN classifier, with FGPL in the first projection when enabled |

## Installation

The environment file targets Linux x86-64 with an NVIDIA GPU and a driver
compatible with the PyTorch CUDA 12.4 build. Its core versions come from the
project's Conda package list.

| Component | Version |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.5.1 (CUDA 12.4) |
| PyTorch Geometric | 2.6.1 |
| NumPy / SciPy / scikit-learn | 1.26.4 / 1.13.1 / 1.5.2 |
| DeepRobust / NetworkX / Loguru | 0.2.11 / 3.4.2 / 0.7.3 |

From the repository root, create the environment and check that the training
entry point can load its dependencies.

```bash
conda env create -f environment.yml
conda activate fracpic-gcn
python -m pip check
python train.py --help
```

The first training run downloads the selected dataset into `data/` if needed.
It also computes and caches a Personalized PageRank (PPR) matrix when the
corresponding cache file is absent. First-time downloads require network access.

## Run FracPIC-GCN

Run the preflight check, then start the full experiment suite from the repository
root. The examples select GPU index `0`. Use `--gpu 1` to select the second GPU.

```bash
python run_fracpic_gcn.py --gpu 0 --dry-run

python run_fracpic_gcn.py --gpu 0 --max-retries 1
```

The dry run checks CUDA availability, logs the detected environment, and prints
the 12 settings. It can report absent PPR caches before the first run. The full
command invokes `train.py` once for each of the 12 main settings in Table 2,
using five seeds (11 through 15) per setting.

| Dataset | `--dataset` | Labeled fraction | Uniform `alpha / c` | Pair `alpha / c` |
| --- | --- | ---: | ---: | ---: |
| Coauthor CS | `cs` | 1% | 0.17 / 1.262 | 0.34 / 1.0 |
| Amazon Photo | `photo` | 1% | 0.30 / 2.50 | 0.406 / 1.188 |
| Cora | `cora` | 5% | 0.72 / 0.75 | 0.702 / 1.284 |
| Pubmed | `pubmed` | 1% | 0.808 / 1.0 | 0.72 / 0.50 |
| DBLP | `dblp` | 1% | 0.60 / 0.952 | 0.721 / 1.0 |
| Citeseer | `citeseer` | 5% | 0.20 / 0.30 | 0.17 / 0.751 |

All settings use a label noise rate of 20%. The runner records the remaining
training arguments for each setting.

### Single-run example

This command runs Cora with uniform noise using one seed.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset cora --noise uniform --ptb_rate 0.2 --label_rate 0.05 \
  --K 50 --local_conflict_weight 0.9 --warmup_epochs 30 --scale1 1.0 \
  --temp 0.01 --alpha 0.72 --c 0.75 --runs 1
```

### Results and resuming

The runner writes to `fracpic_gcn_results/` by default. It records the environment
and run progress in `runner.log` and saves one training log per attempt.
The training console and logs identify the method and report test accuracy
for each seed, followed by its mean and standard deviation across seeds.
The training entry point also writes logs under `logs/fracpic_gcn/<dataset>/`.
Accuracies are reported as fractions in the logs and CSV.

| File | Contents |
| --- | --- |
| `state.json` | Checkpoint tracking completed experiment invocations |
| `results.csv` | One row per completed setting with the method name, mean, standard deviation, and five per-seed test accuracies |
| `results.txt` | Text summary of experiment results and progress |

To resume, run the full command again with the same output directory. The
runner skips completed invocations and restarts an interrupted invocation.
For a separate run, specify a new directory with `--output-dir`.
Use a fresh output directory with this version if you have a v2 checkpoint.

## Run the ICGNN baseline

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

This runs the same 12 dataset and noise settings once, with five seeds per
setting. It uses `--alpha 1.0` to select the first-order GCN projection.
The console identifies this mode as `ICGNN baseline`, and its training logs
are written under `logs/icgnn_baseline/<dataset>/`.

## Acknowledgments

This implementation builds on [ICGNN](https://github.com/wayc04/ICGNN) by
Wei Ju, Wei Zhang, Siyu Yi, Zhengyang Mao, Yifan Wang, Jingyuan Yuan,
Zhiping Xiao, Ziyue Qiao, and Ming Zhang, *Identifying and Correcting Label
Noise for Robust GNNs via Influence Contradiction* (ICML 2026).
The upstream implementation also acknowledges NRGNN as a reference.

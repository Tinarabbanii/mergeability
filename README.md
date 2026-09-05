# Can Mergeability Be Predicted Data-Free?
**_And is k-way merging additive in the tasks?_**

Fine-tuned models can be merged by averaging their weights, sometimes for free, sometimes catastrophically. *Demystifying Mergeability* predicts which, but its strongest signal is **gradient alignment**, which needs calibration data. We ask what survives with **only the weights**, and whether pairwise mergeability extends to groups.

📄 **Target paper:** [Demystifying Mergeability](https://arxiv.org/abs/2601.22285) (Zhou, Zhao, Yu, Rodolà, ICML 2026). Both questions are open problems named in its Appendix A.1.

## Repository
core modules:

```
src/
  utils.py       state_dict algebra: task vectors, flatten, subsets
  config.py      configs/*.yaml -> one Config object
  backends.py    where models come from: synthetic | CLIP
  metrics/       geometry, rank, subspace (data-free) + functional (not)
  merging.py     weight averaging, task arithmetic, TIES
  pipeline.py    metrics + merges + accuracies -> two CSVs
  predict.py     L1 fit, leave-one-task-out CV, nulls, bootstrap
  experiments/   e0..e5
  viz.py         figure generation
scripts/         one runner per experiment
configs/         the experimental setup lives here
artifacts/       cached results
figures/
```

Every function takes a **list** of task vectors, never a pair, which is what makes the k-way question more efficient to answer.

## Run

```bash
uv sync
```

```bash
uv run python scripts/run_e0.py     # sanity: checkpoints load, fine-tuning helped
```
```bash
uv run python scripts/run_e1.py     # ground truth: metrics + merges + accuracies
```
```bash
uv run python scripts/run_e2.py     # RQ1: data-free vs full predictor
```
```bash
uv run python scripts/run_e3.py     # which metrics matter, and is that reproducible
```
```bash
uv run python scripts/run_e4.py     # RQ2: does pairwise predict k-way?
```
```bash
uv run python scripts/run_e5.py     # null baselines + bootstrap intervals
```
```bash
uv run python scripts/make_figures.py
```

Items 3 and 4 (TIES density sweep, calibration size):

```bash
uv run python scripts/run_density_sweep.py --backend clip --k 2
```
```bash
uv run python scripts/analyze_density_sweep.py --backend clip
```
```bash
uv run python scripts/analyze_calibration.py --backend clip --k 2
```

Or everything at once -- e0-e5 at every k, the density sweep, the calibration
study and all ten figures:

```bash
uv run python scripts/run_all.py                  # synthetic, ~90 s
uv run python scripts/run_all.py --backend clip   # the real benchmark
```

Add `--quick` to skip the density sweep and the calibration study.

## Getting the checkpoints

The CLIP backend needs eight fine-tuned ViT-B/32 encoders plus the zero-shot
model, from the Task Arithmetic release:

<https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw>

Download the `ViT-B-32` folder. The files are pickled `ImageEncoder` objects, so
convert them to plain state dicts once:

```bash
uv run python scripts/download_checkpoints.py --list
```
```bash
uv run python scripts/download_checkpoints.py --convert /path/to/ViT-B-32
```

Test images are fetched automatically: MNIST, SVHN, GTSRB, EuroSAT and DTD from
torchvision, RESISC45, Cars and SUN397 from the `tanganke/*` HuggingFace mirrors.

## Two backends

`--backend synthetic` generates tiny models and runs the whole pipeline in seconds, so every stage is verified before any GPU time is spent. It is the **test harness, not a source of findings**.

`--backend clip` is the experiment: CLIP ViT-B/32 encoders fine-tuned on image classification tasks, released by [Task Arithmetic](https://github.com/mlfoundations/task_vectors).

## Reading the numbers

With this few task pairs, the fitting procedure alone reaches r ≈ 0.5 on pure noise. **Every reported correlation carries two null baselines and a bootstrap interval**, a bare correlation is not interpretable here.

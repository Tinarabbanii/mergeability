# Can Mergeability Be Predicted Without Data?
**_And does pairwise mergeability carry over to groups?_**

![](./animation.gif)

Fine-tuned models can be merged by averaging their weights, sometimes for free and sometimes catastrophically. *Demystifying Mergeability* predicts which, but its strongest signal is gradient alignment, which needs calibration data. This asks what survives using only the weights, and whether mergeability measured on pairs carries over to groups of three and four.

📄 **Target paper:** [Demystifying Mergeability](https://arxiv.org/abs/2601.22285) (Zhou, Zhao, Yu, Rodolà, ICML 2026) to read the two open problems in Appendix A.1 this takes up.

## Repository
core modules:

```
src/
  utils.py       State-dict algebra and subset enumeration
  config.py      YAML-backed experiment configuration
  backends.py    Model providers: synthetic | CLIP
  metrics/       Data-free (geometry, rank, subspace) + functional
  merging.py     Weight averaging, task arithmetic, TIES
  pipeline.py    Metric computation and merge evaluation
  predict.py     Sparse predictor, leave-one-task-out CV, nulls, bootstrap
  experiments/   e0..e5
  viz.py         Figure generation
scripts/         Experiment runners and analyses
configs/         Experimental setup
artifacts/       Cached results, per backend
figures/
```

Every function takes a **list** of task vectors, never a pair, which is what makes the k-way question cheap to answer.

## Run

```bash
uv sync
```

The synthetic backend runs on CPU and needs nothing else. The two CLIP backends
need checkpoints and datasets, neither redistributed here.

**Datasets.** Eight classification sets. Five resolve through torchvision. Three
do not: the torchvision Stanford Cars mirror is dead, RESISC45 was never in
torchvision, and torchvision's SUN397 is the full 37 GB rather than the split we
score on. The checker prints where to get each one.

```bash
uv run python scripts/check_datasets.py
```

**Checkpoints.** Fine-tuned CLIP encoders from the
[Task Arithmetic release](https://github.com/mlfoundations/task_vectors),
~430 MB each.

```bash
uv run python scripts/download_checkpoints.py --convert /path/to/ViT-B-32
```

Device is picked automatically: CUDA, then MPS, then CPU.

```bash
uv run python scripts/run_all.py                    # synthetic harness, ~7 min
uv run python scripts/run_all.py --backend clip     # ViT-B/32, the benchmark
uv run python scripts/run_all.py --backend clip16   # ViT-B/16, the replication
```

Each backend writes to its own `artifacts/<backend>/`, and every analysis refuses
to load results produced under a different backend or task count. Add `--quick`
to skip the density sweep, the calibration study and the statistical checks.

<details>
<summary>or step by step</summary>

```bash
uv run python scripts/run_e0.py --backend clip   # sanity: checkpoints load, fine-tuning helped
uv run python scripts/run_e1.py --backend clip   # ground truth: metrics + merges + accuracies
uv run python scripts/run_e2.py --backend clip   # RQ1: data-free vs full predictor
uv run python scripts/run_e3.py --backend clip   # which metrics matter, and is that reproducible
uv run python scripts/run_e4.py --backend clip   # RQ2: does pairwise predict k-way?
uv run python scripts/run_e5.py --backend clip   # null baselines + bootstrap intervals
uv run python scripts/make_figures.py --backend clip
```

```bash
# item 3 and item 4: TIES density sweep, calibration size
uv run python scripts/run_density_sweep.py --backend clip --k 2
uv run python scripts/analyze_density_sweep.py --backend clip
uv run python scripts/analyze_calibration.py --backend clip --k 2
```

```bash
# the figures as they appear in the report
uv run python scripts/make_report_figs.py
```

```bash
# the checks behind every reported claim
for c in global_test paired_bootstrap univariate nmatched pooled_k increment_null; do
    uv run python scripts/analyze_$c.py --backend clip
done
```

</details>

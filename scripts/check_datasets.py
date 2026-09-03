from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ROOT, load_config  # noqa: E402

PLAN = {
    "MNIST":    ("MNIST",        dict(train=False),      10, True,  ""),
    "SVHN":     ("SVHN",         dict(split="test"),     65, True,  ""),
    "GTSRB":    ("GTSRB",        dict(split="test"),     90, True,  ""),
    "EuroSAT":  ("EuroSAT",      dict(),                 90, True,  "no train/test split; one pool"),
    "DTD":      ("DTD",          dict(split="test"),    600, True,  "archive contains all splits"),
    "Cars":     ("StanfordCars", dict(split="test"),   1900, False, "torchvision URL IS DEAD -- Kaggle/HF"),
    "SUN397":   ("SUN397",       dict(),              37000, False, "use the tanganke/sun397 HF mirror, not torchvision"),
    "RESISC45": (None,           dict(),                350, False, "NOT in torchvision -- Kaggle/HF only"),
}
ALTERNATIVES = """
Sourcing the three hard ones
RESISC45   Kaggle: search "RESISC45" and attach the dataset to your notebook
           HuggingFace: `timm/resisc45` or `blanchon/RESISC45`
Cars       Kaggle: "Stanford Cars Dataset" (the torchvision mirror is dead)
SUN397     HuggingFace: `tanganke/sun397` (torchvision wants the full 37 GB).

On Kaggle you ATTACH a dataset rather than downloading it -- it appears under
/kaggle/input and costs you none of your own disk. That is the single best
reason to run the real experiments there rather than locally.

Whatever you stage, add a branch for it in src/clip_assets.py:build_loader so
the pipeline can find it.
"""

def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true",
                    help="actually download (skips the ones marked unavailable)")
    ap.add_argument("--only", nargs="*", help="restrict to these task names")
    args = ap.parse_args()

    cfg = load_config("clip")
    data_root = ROOT / "data"
    data_root.mkdir(exist_ok=True)
    configured = list(cfg.task_names)

    print(f"configured tasks: {configured}")
    print(f"free disk: {free_gb(data_root):.1f} GB\n")
    print(f"{'task':<10}{'in config':<11}{'torchvision':<13}{'~MB':>7}  note")
    print("-" * 78)

    easy, hard = [], []
    for name, (cls_name, kwargs, mb, usable, note) in PLAN.items():
        (easy if usable else hard).append(name)
        print(f"{name:<10}{'yes' if name in configured else 'no':<11}"
              f"{(cls_name or 'NOT PRESENT'):<13}{mb:>7}  {note}")

    print(f"\nstage-able with one line: {easy}  ->  "
          f"{len(easy)} tasks = {len(easy) * (len(easy) - 1) // 2} pairs")
    print(f"needs manual sourcing:    {hard}")
    print(ALTERNATIVES)

    if not args.download:
        print("Report only. Re-run with --download to fetch the easy ones.")
        return

    targets = [t for t in easy if not args.only or t in args.only]
    total_mb = sum(PLAN[t][2] for t in targets)
    if total_mb / 1000 > free_gb(data_root) - 1.0:
        print(f"REFUSING: {total_mb} MB needed, only {free_gb(data_root):.1f} GB free. "
              f"Free up space or use --only to fetch fewer.")
        return

    import torchvision.datasets as tvd
    print(f"\ndownloading {targets}  (~{total_mb} MB)\n")
    for name in targets:
        cls_name, kwargs, mb, _, _ = PLAN[name]
        try:
            ds = getattr(tvd, cls_name)(root=str(data_root), download=True, **kwargs)
            print(f"  ok    {name:<10} {len(ds):>7} images")
        except Exception as exc:
            print(f"  FAIL  {name:<10} {type(exc).__name__}: {str(exc)[:90]}")

    print(f"\nfree disk now: {free_gb(data_root):.1f} GB")
    print("Next: edit configs/tasks.yaml:clip.tasks to exactly what you staged,")
    print("then run  python scripts/run_e0.py --backend clip")
if __name__ == "__main__":
    main()

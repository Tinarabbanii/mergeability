import argparse, re, shutil, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.backends import get_backend
from src.config import load_config
from src.experiments import (e0_sanity, e1_pairwise, e2_datafree, e3_importance, e4_kway, e5_robustness)
from src.utils import set_seed
from src.viz import make_all

ROOT = Path(__file__).resolve().parent.parent

def _script(name: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / name), *args]
    print("\n$ " + " ".join(cmd[1:]))
    subprocess.run(cmd, check=True, cwd=ROOT)

def _calibration(cfg, backend, ks) -> None:
    from src.pipeline import build_metrics
    p = ROOT / "configs" / "metrics.yaml"
    original = p.read_text()
    cal10 = cfg.artifact("metrics_cal10.csv")
    shutil.copy(cfg.artifact("metrics.csv"), cal10)
    try:
        patched = re.sub(r"samples_per_task:\s*10\b", "samples_per_task: 100", original)
        if patched == original:
            print("  calibration skipped: samples_per_task is not 10")
            return
        p.write_text(patched)
        build_metrics(load_config(cfg.backend), backend)
        shutil.copy(cfg.artifact("metrics.csv"), cfg.artifact("metrics_cal100.csv"))
    finally:
        p.write_text(original)
        shutil.copy(cal10, cfg.artifact("metrics.csv"))
    for k in ks:
        _script("analyze_calibration.py", "--backend", cfg.backend, "--k", str(k))

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="synthetic", choices=["synthetic", "clip", "clip16"])
    ap.add_argument("--skip-e1", action="store_true",
                    help="reuse cached metrics.csv / results.csv")
    ap.add_argument("--quick", action="store_true",
                    help="e0-e5 only: skip the density sweep and the calibration study")
    args = ap.parse_args()

    cfg = load_config(args.backend)
    set_seed(cfg.seed)
    backend = get_backend(cfg)
    ks = list(cfg.k_values)
    t0 = time.time()

    print("\n" + "=" * 74); e0_sanity.run(cfg, backend)
    print("\n" + "=" * 74); e1_pairwise.run(cfg, backend, args.skip_e1, args.skip_e1)
    print("\n" + "=" * 74); e2_datafree.run(cfg, backend)
    for k in ks:
        print("\n" + "=" * 74); e3_importance.run(cfg, backend, k)
    print("\n" + "=" * 74); e4_kway.run(cfg, backend)
    for k in ks:
        print("\n" + "=" * 74); e5_robustness.run(cfg, backend, k)
    for k in ks:
        print("\n" + "=" * 74); e3_importance.run(cfg, backend, k)

    if not args.quick:
        print("\n" + "=" * 74)
        _script("run_density_sweep.py", "--backend", args.backend, "--k", "2")
        _script("analyze_density_sweep.py", "--backend", args.backend)
        print("\n" + "=" * 74)
        _calibration(cfg, backend, ks)

    print("\n" + "=" * 74)
    make_all(cfg)
    from src.predict import fallback_report
    print("\n" + "=" * 74)
    print(fallback_report())
    print(f"\ntotal: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()

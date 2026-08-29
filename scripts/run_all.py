import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.backends import get_backend
from src.config import load_config
from src.experiments import (e0_sanity, e1_pairwise, e2_datafree, e3_importance, e4_kway, e5_robustness)
from src.utils import set_seed
from src.viz import make_all

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="synthetic", choices=["synthetic", "clip"])
    ap.add_argument("--skip-e1", action="store_true",
                    help="reuse cached metrics.csv / results.csv")
    args = ap.parse_args()

    cfg = load_config(args.backend)
    set_seed(cfg.seed)
    backend = get_backend(cfg)
    t0 = time.time()

    for name, fn in [
        ("e0", lambda: e0_sanity.run(cfg, backend)),
        ("e1", lambda: e1_pairwise.run(cfg, backend, args.skip_e1, args.skip_e1)),
        ("e2", lambda: e2_datafree.run(cfg, backend)),
        ("e3", lambda: e3_importance.run(cfg, backend)),
        ("e4", lambda: e4_kway.run(cfg, backend)),
        ("e5", lambda: e5_robustness.run(cfg, backend)),
    ]:
        print("\n" + "=" * 74)
        fn()
    print("\n" + "=" * 74)
    make_all(cfg)
    print(f"\ntotal: {time.time() - t0:.1f}s")
if __name__ == "__main__":
    main()

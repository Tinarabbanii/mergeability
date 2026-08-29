import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.backends import get_backend
from src.config import load_config
from src.experiments import e5_robustness
from src.utils import set_seed

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="synthetic", choices=["synthetic", "clip"])
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()
    cfg = load_config(args.backend)
    set_seed(cfg.seed)
    backend = get_backend(cfg)
    e5_robustness.run(cfg, backend, args.k)

if __name__ == "__main__":
    main()

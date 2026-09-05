import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.backends import get_backend
from src.config import load_config
from src.experiments import e0_sanity
from src.utils import set_seed

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="synthetic", choices=["synthetic", "clip", "clip16"])
    args = ap.parse_args()
    cfg = load_config(args.backend)
    set_seed(cfg.seed)
    backend = get_backend(cfg)
    e0_sanity.run(cfg, backend)

if __name__ == "__main__":
    main()

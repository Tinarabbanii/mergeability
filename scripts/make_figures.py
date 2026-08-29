import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.viz import make_all

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="synthetic", choices=["synthetic", "clip"])
    args = ap.parse_args()
    make_all(load_config(args.backend))
if __name__ == "__main__":
    main()
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ROOT, load_config  # noqa: E402

TASKS_8 = ["MNIST", "EuroSAT", "GTSRB", "SVHN", "RESISC45", "DTD", "Cars", "SUN397"]

def cmd_list(cfg) -> None:
    spec = cfg.tasks["clip"]
    print("Checkpoints this project expects\n")
    print(f"  model:  {spec['model']}")
    print(f"  source: {spec['checkpoint_url']}")
    print(f"  into:   {ROOT / spec['checkpoint_dir']}\n")
    print("  zeroshot.pt          the pretrained encoder (theta_0)")
    for t in spec["tasks"]:
        print(f"  {t + '.pt':<21}fine-tuned encoder for {t}")
    for t in spec["tasks"]:
        print(f"  head_{t + '.pt':<16}frozen classification head for {t}")
    print("\nAll eight released tasks: " + ", ".join(TASKS_8))
    print("\nSize: roughly 600 MB per encoder. Eight tasks plus the pretrained")
    print("model is about 5.4 GB -- do this on Kaggle, not on a laptop.")

_FILLER = {"head", "model", "checkpoint", "finetuned", "final", "best"}
_KNOWN = ["RESISC45", "EuroSAT", "SUN397", "ImageNet", "GTSRB", "MNIST",
          "SVHN", "Cars", "DTD"]

def _normalise_name(path: Path, src_dir: Path) -> str:
    rel = path.relative_to(src_dir)
    is_head = any(part.lower() == "head" for part in rel.parts[:-1]) \
        or path.stem.lower().startswith("head")

    squashed = re.sub(r"[^a-z0-9]", "", path.stem.lower())
    if squashed in ("zeroshot", "pretrained", "zeroshotmodel"):
        return "zeroshot.pt"

    if path.stem.lower() in ("finetuned", "checkpoint") and len(rel.parts) > 1:
        return f"{rel.parts[-2]}.pt"
    tokens = [t for t in re.split(r"[\s_\-]+", path.stem) if t.lower() not in _FILLER]
    task = " ".join(tokens).strip()
    for known in _KNOWN:
        if known.lower() == task.lower().replace(" ", ""):
            task = known
            break
    if not task:
        task = rel.parts[-2] if len(rel.parts) > 1 else path.stem
    return f"head_{task}.pt" if is_head else f"{task}.pt"

def _install_stubs() -> None:
    import types
    import torch.nn as nn
    class _Stub(nn.Module):
        def __init__(self, *a, **k):
            super().__init__()
    for name in ("src", "src.modeling"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["src.modeling"].ImageEncoder = _Stub
    return _Stub


def _load_pickled(path: Path):
    import re
    stub = _install_stubs()
    for _ in range(40):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except AttributeError as exc:
            m = re.search(r"Can't get attribute '([^']+)' on <module '([^']+)'", str(exc))
            if not m:
                raise
            attr, mod = m.group(1), m.group(2)
            if mod not in sys.modules:
                raise
            setattr(sys.modules[mod], attr, stub)
    raise RuntimeError(f"too many missing classes while loading {path}")

def _to_state_dict(obj) -> dict:
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    if not isinstance(obj, dict):
        raise TypeError(f"cannot interpret {type(obj)} as a state_dict")
    for key in ("state_dict", "model", "image_encoder"):
        if key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    return {k: v.detach().cpu().float()
            for k, v in obj.items() if torch.is_tensor(v) and v.is_floating_point()}
def cmd_convert(cfg, src_dir: Path) -> None:
    out_dir = ROOT / cfg.tasks["clip"]["checkpoint_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(src_dir)

    found = sorted(src_dir.rglob("*.pt")) + sorted(src_dir.rglob("*.pth"))
    if not found:
        raise SystemExit(f"no .pt/.pth files under {src_dir}")
    print(f"converting {len(found)} files\n  from {src_dir}\n  into {out_dir}\n")
    ok = fail = 0
    for path in found:
        target_name = _normalise_name(path, src_dir)
        try:
            obj = _load_pickled(path)
            sd = _to_state_dict(obj)
            if not sd:
                raise ValueError("no float tensors found")
            torch.save(sd, out_dir / target_name)
            n = sum(v.numel() for v in sd.values())
            print(f"  ok    {str(path.relative_to(src_dir)):<28} -> {target_name:<16}"
                  f"{len(sd):>4} tensors  {n / 1e6:>6.1f}M")
            ok += 1
        except Exception as exc:
            print(f"  FAIL  {str(path.relative_to(src_dir)):<28} "
                  f"{type(exc).__name__}: {str(exc)[:70]}")
            fail += 1
    if not (out_dir / "zeroshot.pt").exists():
        print("\n  *** WARNING: no zeroshot.pt produced. ***")
        print("  The pretrained checkpoint is REQUIRED -- a task vector is")
        print("  finetuned MINUS pretrained. Go back to the Drive folder and")
        print("  fetch zeroshot.pt from the top level of ViT-B-32/.")
        print("  Substituting vanilla OpenAI CLIP does NOT work: its text tower")
        print("  differs from the true base by 0.073 in norm, even though")
        print("  fine-tuning never touches the text tower, so every task vector")
        print("  would carry the same systematic error.")
    print(f"\n{ok} converted, {fail} failed -> {out_dir}")
    if fail:
        print("\nIf the failures say ModuleNotFoundError, you are not running from a")
        print("directory where the task_vectors repo's `src` package is importable.")
        print("cd into their repo checkout and run this script by absolute path.")
    else:
        print("\nNext: python scripts/run_e0.py --backend clip")
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show what is expected")
    ap.add_argument("--convert", metavar="DIR",
                    help="convert pickled checkpoints in DIR to plain state_dicts")
    args = ap.parse_args()
    cfg = load_config("clip")
    if args.convert:
        cmd_convert(cfg, Path(args.convert))
    else:
        cmd_list(cfg)
if __name__ == "__main__":
    main()

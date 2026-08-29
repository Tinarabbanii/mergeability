from __future__ import annotations
import pandas as pd
from ..config import Config
from ..metrics import MetricComputer
from ..utils import sd_flatten

def run(cfg: Config, backend) -> pd.DataFrame:
    print(f"e0: sanity check  [backend={cfg.backend}]")
    tasks = cfg.task_names
    mc = MetricComputer(cfg, backend)

    theta_pre = backend.pretrained()
    print(f"  pretrained: {len(theta_pre)} tensors, "
          f"{sum(v.numel() for v in theta_pre.values()):,} parameters")

    rows = []
    for task in tasks:
        ft = backend.finetuned(task)
        missing = set(theta_pre) - set(ft)
        if missing:
            raise RuntimeError(f"{task}: {len(missing)} keys missing vs pretrained, "
                               f"e.g. {sorted(missing)[:3]}")

        tv_norm = sd_flatten(mc.task_vector(task)).norm().item()
        if tv_norm < 1e-8:
            raise RuntimeError(f"{task}: task vector is zero -- checkpoint identical "
                               f"to pretrained, or subtracted the wrong way round")

        own = backend.evaluate(ft, task)
        base = backend.evaluate(theta_pre, task)
        rows.append({"task": task, "expert_accuracy": own,
                     "pretrained_accuracy": base, "task_vector_norm": tv_norm})
        flag = "" if own > base else "   <-- FINE-TUNING DID NOT HELP"
        print(f"  {task:<10} acc={own:.4f}  (pretrained {base:.4f}){flag}   "
              f"||tau||={tv_norm:.4f}")
    df = pd.DataFrame(rows)

    print("\n  cross-task accuracy matrix (rows = expert, cols = evaluated on):")
    header = "         " + "".join(f"{t:>10}" for t in tasks)
    print(header)
    diag_wins = 0
    for src in tasks:
        ft = backend.finetuned(src)
        accs = [backend.evaluate(ft, dst) for dst in tasks]
        own_acc = accs[tasks.index(src)]
        if own_acc >= max(accs) - 1e-6:
            diag_wins += 1
        print(f"  {src:<8}" + "".join(f"{a:>10.3f}" for a in accs))

    print(f"\n  diagonal dominance (informational): {diag_wins}/{len(tasks)}")

    improved = int((df.expert_accuracy > df.pretrained_accuracy).sum())
    print(f"  fine-tuning improved over pretrained: {improved}/{len(tasks)}")
    if improved < len(tasks):
        raise RuntimeError(
            "at least one fine-tuned checkpoint is no better than the pretrained "
            "model on its own task -- the checkpoints are wrong or mislabelled"
        )
    print("  PASS: setup is sound.")
    df.to_csv(cfg.artifact("e0_sanity.csv"), index=False)
    print(f"  -> {cfg.artifact('e0_sanity.csv')}")
    return df

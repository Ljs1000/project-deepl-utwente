import yaml
import csv
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from .data import build_loaders
from .models_cnn import PneumoniaCNN
from .attacks import pgd_attack
from .utils import get_device


def main():
    cfg_path = Path("config.yaml").resolve()
    print("Using config:", cfg_path)

    cfg = yaml.safe_load(open(cfg_path, "r"))
    device = get_device()

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "pgd_attack.csv"

    _train_loader, _val_loader, test_loader, _ = build_loaders(
        cfg["data_dir"],
        int(cfg["img_size"]),
        int(cfg["batch_size"]),
        int(cfg["num_workers"]),
        augment_train=False,
    )

    model = PneumoniaCNN().to(device)
    ckpt = torch.load("classifier_best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss()

    # PGD params (defaults if not in config)
    pgd_cfg = cfg.get("pgd", {})
    alpha = float(pgd_cfg.get("alpha", 0.002))
    steps = int(pgd_cfg.get("steps", 10))
    random_start = bool(pgd_cfg.get("random_start", True))

    rows = []
    for eps in cfg["attack"]["epsilons"]:
        eps = float(eps)
        if eps == 0.0:
            continue

        ys, yps = [], []
        for x, y in tqdm(test_loader, desc=f"PGD eps={eps} (a={alpha}, steps={steps})"):
            x = x.to(device)
            y = y.float().to(device)

            x_adv = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=steps,
                               loss_fn=loss_fn, random_start=random_start)

            with torch.no_grad():
                logits = model(x_adv).squeeze(1)
                prob = torch.sigmoid(logits)

            ys.append(y.cpu())
            yps.append(prob.cpu())

        y_true = torch.cat(ys).numpy()
        y_prob = torch.cat(yps).numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)

        print(f"eps={eps:>7}: pgd_acc={acc:.4f}")
        rows.append({"epsilon": eps, "alpha": alpha, "steps": steps, "pgd_acc": float(acc)})

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epsilon", "alpha", "steps", "pgd_acc"])
        w.writeheader()
        w.writerows(rows)

    print("Wrote:", out_csv.resolve())


if __name__ == "__main__":
    main()

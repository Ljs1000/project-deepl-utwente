import yaml
import csv
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from .data import build_loaders
from .models_cnn import PneumoniaCNN
from .attacks import fgsm_attack
from .utils import get_device


def main():
    cfg_path = Path("config.yaml").resolve()
    print("Using config:", cfg_path)

    cfg = yaml.safe_load(open(cfg_path, "r"))
    device = get_device()

    # Ensure output dirs exist
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "fgsm_attack.csv"

    _train_loader, _val_loader, test_loader, _ = build_loaders(
        cfg["data_dir"],
        int(cfg["img_size"]),
        int(cfg["batch_size"]),
        int(cfg["num_workers"]),
        augment_train=False,  # clean transforms for eval
    )

    model = PneumoniaCNN().to(device)
    ckpt = torch.load("classifier_best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss()

    rows = []
    for eps in cfg["attack"]["epsilons"]:
        eps = float(eps)

        ys, yps = [], []
        for x, y in tqdm(test_loader, desc=f"FGSM eps={eps}"):
            x = x.to(device)
            y = y.float().to(device)

            x_adv = fgsm_attack(model, x, y, eps, loss_fn)
            with torch.no_grad():
                logits = model(x_adv).squeeze(1)
                prob = torch.sigmoid(logits)

            ys.append(y.cpu())
            yps.append(prob.cpu())

        y_true = torch.cat(ys).numpy()
        y_prob = torch.cat(yps).numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)

        print(f"eps={eps:>7}: adv_acc={acc:.4f}")
        rows.append({"epsilon": eps, "adv_acc": float(acc)})

    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epsilon", "adv_acc"])
        w.writeheader()
        w.writerows(rows)

    print("Wrote:", out_csv.resolve())


if __name__ == "__main__":
    main()

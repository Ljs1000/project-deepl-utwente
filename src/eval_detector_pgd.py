import yaml
import csv
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .data import build_loaders
from .models_vae import VAE
from .models_cnn import PneumoniaCNN
from .attacks import pgd_attack
from .utils import get_device


def laplacian(x: torch.Tensor) -> torch.Tensor:
    k = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
        dtype=x.dtype,
        device=x.device
    ).view(1, 1, 3, 3)
    return F.conv2d(x, k, padding=1)


def recon_error_laplacian_l1(vae, x):
    vae.eval()
    with torch.no_grad():
        x_hat, _, _ = vae(x)
        if x_hat.shape[-2:] != x.shape[-2:]:
            x_hat = F.interpolate(x_hat, size=x.shape[-2:], mode="bilinear", align_corners=False)

        lx = laplacian(x)
        lxh = laplacian(x_hat)
        err = (lxh - lx).abs().mean(dim=(1, 2, 3))
    return err


def collect_errors_by_class(vae, loader, device):
    errs = {0: [], 1: []}
    for x, y in loader:
        x = x.to(device)
        y_np = y.numpy()
        e = recon_error_laplacian_l1(vae, x).cpu().numpy()
        for cls in (0, 1):
            m = (y_np == cls)
            if m.any():
                errs[cls].append(e[m])
    return {k: np.concatenate(v) if len(v) else np.array([]) for k, v in errs.items()}


def threshold_by_target_fpr(clean_errs_cls: np.ndarray, target_fpr: float):
    q = 100.0 * (1.0 - target_fpr)
    return float(np.percentile(clean_errs_cls, q))


def main():
    cfg_path = Path("config.yaml").resolve()
    print("Using config:", cfg_path)

    cfg = yaml.safe_load(open(cfg_path, "r"))
    device = get_device()

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "detector_laplacian_pgd.csv"

    train_loader, _val_loader, test_loader, class_to_idx = build_loaders(
        cfg["data_dir"],
        int(cfg["img_size"]),
        int(cfg["batch_size"]),
        int(cfg["num_workers"]),
        augment_train=False,
    )
    print("class_to_idx:", class_to_idx)

    # Load VAE (same VAE!)
    latent_dim = int(cfg["vae"]["latent_dim"])
    vae = VAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load("vae_last.pt", map_location=device)["vae"])

    # Threshold calibrated on TRAIN clean
    target_fpr = 0.05
    train_errs = collect_errors_by_class(vae, train_loader, device)
    thr = {cls: threshold_by_target_fpr(train_errs[cls], target_fpr) for cls in (0, 1)}
    print(
        f"TRAIN thresholds (target FPR={target_fpr:.2%}, Laplacian-L1): "
        f"thr_normal={thr[0]:.6f}, thr_pneumonia={thr[1]:.6f}"
    )

    # Classifier for PGD generation
    clf = PneumoniaCNN().to(device)
    ckpt = torch.load("classifier_best.pt", map_location=device)
    clf.load_state_dict(ckpt["model"])
    clf.eval()

    import torch.nn as nn
    loss_fn = nn.BCEWithLogitsLoss()

    pgd_cfg = cfg.get("pgd", {})
    alpha = float(pgd_cfg.get("alpha", 0.002))
    steps = int(pgd_cfg.get("steps", 10))
    random_start = bool(pgd_cfg.get("random_start", True))

    eps_list = [float(e) for e in cfg["attack"]["epsilons"] if float(e) > 0.0]

    rows = []
    for eps in eps_list:
        scores, labels, y_true_all = [], [], []

        for x, y in test_loader:
            x = x.to(device)
            y_np = y.numpy()
            y_t = torch.tensor(y_np, device=device).float()

            err_clean = recon_error_laplacian_l1(vae, x).cpu().numpy()

            x_adv = pgd_attack(
                clf, x, y_t, eps=eps, alpha=alpha, steps=steps,
                loss_fn=loss_fn, random_start=random_start
            )
            err_adv = recon_error_laplacian_l1(vae, x_adv).cpu().numpy()

            scores.append(np.concatenate([err_clean, err_adv]))
            labels.append(np.concatenate([np.zeros_like(err_clean), np.ones_like(err_adv)]))
            y_true_all.append(np.concatenate([y_np, y_np]))

        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
        y_true_all = np.concatenate(y_true_all)

        auc = float(roc_auc_score(labels, scores))

        pred_adv = np.zeros_like(labels, dtype=int)
        for cls in (0, 1):
            m = (y_true_all == cls)
            pred_adv[m] = (scores[m] >= thr[cls]).astype(int)

        tpr = float((pred_adv[labels == 1] == 1).mean())
        fpr = float((pred_adv[labels == 0] == 1).mean())

        print(f"eps={eps:>7}: PGD-detector AUC={auc:.4f} | TPR={tpr:.3f} FPR={fpr:.3f}")
        rows.append({"epsilon": eps, "alpha": alpha, "steps": steps, "auc": auc, "tpr": tpr, "fpr": fpr})

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epsilon", "alpha", "steps", "auc", "tpr", "fpr"])
        w.writeheader()
        w.writerows(rows)

    print("Wrote:", out_csv.resolve())


if __name__ == "__main__":
    main()

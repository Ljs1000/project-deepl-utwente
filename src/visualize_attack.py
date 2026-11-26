import os
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from .data import build_loaders
from .models_cnn import PneumoniaCNN
from .attacks import fgsm_attack
from .utils import get_device, seed_everything


def laplacian(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N,1,H,W)
    Returns Laplacian-filtered x (high-frequency emphasis).
    """
    k = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
        dtype=x.dtype,
        device=x.device
    ).view(1, 1, 3, 3)
    return F.conv2d(x, k, padding=1)


def to_np_img(x: torch.Tensor) -> np.ndarray:
    """(1,H,W) tensor in [0,1] -> (H,W) numpy"""
    return x.detach().cpu().squeeze(0).numpy()


def save_comparison_figure(
    out_path: Path,
    x_clean_1hw: torch.Tensor,
    x_adv_1hw: torch.Tensor,
    title: str,
    show_laplacian: bool = True,
):
    """
    Saves a figure showing:
      Clean | Adversarial | |Delta| (+ optionally Laplacian|Delta|)
    """
    clean = to_np_img(x_clean_1hw)
    adv = to_np_img(x_adv_1hw)

    delta = np.abs(adv - clean)

    # Normalize delta for visualization (not for metrics)
    delta_norm = delta / (delta.max() + 1e-12)

    panels = 4 if show_laplacian else 3
    fig, axes = plt.subplots(1, panels, figsize=(4 * panels, 4))
    fig.suptitle(title, fontsize=12)

    axes[0].imshow(clean, cmap="gray")
    axes[0].set_title("Clean")
    axes[0].axis("off")

    axes[1].imshow(adv, cmap="gray")
    axes[1].set_title("Adversarial (FGSM)")
    axes[1].axis("off")

    axes[2].imshow(delta_norm, cmap="gray")
    axes[2].set_title("|Δ| (normalized)")
    axes[2].axis("off")

    if show_laplacian:
        # Laplacian difference emphasizes high-frequency perturbations
        clean_t = x_clean_1hw.unsqueeze(0)  # (1,1,H,W)
        adv_t = x_adv_1hw.unsqueeze(0)
        lap_clean = laplacian(clean_t).squeeze().detach().cpu().numpy()
        lap_adv = laplacian(adv_t).squeeze().detach().cpu().numpy()
        lap_delta = np.abs(lap_adv - lap_clean)
        lap_delta_norm = lap_delta / (lap_delta.max() + 1e-12)

        axes[3].imshow(lap_delta_norm, cmap="gray")
        axes[3].set_title("|Δ Laplacian| (normalized)")
        axes[3].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    seed_everything(42)

    cfg_path = Path("config.yaml").resolve()
    cfg = yaml.safe_load(open(cfg_path, "r"))

    device = get_device()
    print("Device:", device)

    # Use clean transforms for visualization (no augmentation)
    _train_loader, _val_loader, test_loader, class_to_idx = build_loaders(
        cfg["data_dir"],
        int(cfg["img_size"]),
        int(cfg["batch_size"]),
        int(cfg["num_workers"]),
        augment_train=False,
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Load classifier checkpoint
    model = PneumoniaCNN().to(device)
    ckpt = torch.load("classifier_best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss()

    # Epsilons to visualize (you can edit)
    eps_list = [0.002, 0.01]

    out_dir = Path("outputs/attack_viz")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Saving figures to:", out_dir.resolve())

    # Collect a few samples
    n_per_eps = 6  # saves 6 images per epsilon
    saved = {eps: 0 for eps in eps_list}

    with torch.no_grad():
        for x, y in test_loader:
            # x: (N,1,H,W), y: (N,)
            # We'll generate adversarial per epsilon (needs grad), so do that outside no_grad
            pass

    # We need gradients for FGSM, so loop normally
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device).float()

        # Get model predictions on clean
        with torch.no_grad():
            logits_clean = model(x).squeeze(1)
            prob_clean = torch.sigmoid(logits_clean)
            pred_clean = (prob_clean >= 0.5).long()

        for eps in eps_list:
            if saved[eps] >= n_per_eps:
                continue

            # Generate adversarial examples for this batch
            x_adv = fgsm_attack(model, x, y, float(eps), loss_fn)

            with torch.no_grad():
                logits_adv = model(x_adv).squeeze(1)
                prob_adv = torch.sigmoid(logits_adv)
                pred_adv = (prob_adv >= 0.5).long()

            # Save a few examples from the batch
            for i in range(x.shape[0]):
                if saved[eps] >= n_per_eps:
                    break

                true_cls = int(y[i].item())
                clean_cls = int(pred_clean[i].item())
                adv_cls = int(pred_adv[i].item())

                title = (
                    f"eps={eps} | true={idx_to_class[true_cls]} | "
                    f"clean_pred={idx_to_class[clean_cls]} (p={prob_clean[i].item():.3f}) | "
                    f"adv_pred={idx_to_class[adv_cls]} (p={prob_adv[i].item():.3f})"
                )

                out_path = out_dir / f"eps_{eps}_sample_{saved[eps]:02d}.png"
                save_comparison_figure(
                    out_path=out_path,
                    x_clean_1hw=x[i].detach().cpu(),
                    x_adv_1hw=x_adv[i].detach().cpu(),
                    title=title,
                    show_laplacian=True,
                )
                saved[eps] += 1

        # Stop once done for all eps
        if all(saved[eps] >= n_per_eps for eps in eps_list):
            break

    print("Done. Saved:", saved)


if __name__ == "__main__":
    main()

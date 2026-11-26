import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path

from .data import build_loaders
from .models_cnn import PneumoniaCNN
from .utils import seed_everything, get_device


def eval_epoch(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.float().to(device)
            logits = model(x).squeeze(1)
            prob = torch.sigmoid(logits)
            ys.append(y.cpu())
            ps.append(prob.cpu())

    y_true = torch.cat(ys).numpy()
    y_prob = torch.cat(ps).numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return acc, auc


def main():
    seed_everything(42)

    cfg_path = Path("config.yaml").resolve()
    print("Using config:", cfg_path)

    cfg = yaml.safe_load(open(cfg_path, "r"))
    device = get_device()

    train_loader, val_loader, test_loader, class_to_idx = build_loaders(
        cfg["data_dir"], int(cfg["img_size"]), int(cfg["batch_size"]), int(cfg["num_workers"])
    )
    print("class_to_idx:", class_to_idx)

    model = PneumoniaCNN().to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    # Robust numeric casting (fixes YAML weirdness)
    raw_lr = cfg["classifier"]["lr"]
    raw_wd = cfg["classifier"]["weight_decay"]
    lr = float(raw_lr)
    wd = float(raw_wd)

    print("Loaded classifier lr type/value:", type(raw_lr), raw_lr)
    print("Loaded classifier wd type/value:", type(raw_wd), raw_wd)
    print("Using classifier lr/wd:", lr, wd)

    opt = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=2, factor=0.3, min_lr=1e-6
    )

    best_val = -1.0
    epochs = int(cfg["classifier"]["epochs"])

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")

        for x, y in pbar:
            x = x.to(device)
            y = y.float().to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

        val_acc, val_auc = eval_epoch(model, val_loader, device)
        scheduler.step(val_acc)
        print(f"val_acc={val_acc:.4f} val_auc={val_auc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {"model": model.state_dict(), "class_to_idx": class_to_idx},
                "classifier_best.pt",
            )

    # final test
    ckpt = torch.load("classifier_best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_acc, test_auc = eval_epoch(model, test_loader, device)
    print(f"TEST: acc={test_acc:.4f} auc={test_auc:.4f}")


if __name__ == "__main__":
    main()

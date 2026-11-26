import yaml
import torch
from tqdm import tqdm
from pathlib import Path

from .data import build_loaders
from .models_vae import VAE, vae_loss
from .utils import seed_everything, get_device


def main():
    seed_everything(42)

    cfg_path = Path("config.yaml").resolve()
    print("Using config:", cfg_path)

    cfg = yaml.safe_load(open(cfg_path, "r"))
    device = get_device()

    # IMPORTANT: augment_train=False for VAE training (clean-only)
    train_loader, _val_loader, _test_loader, _ = build_loaders(
        cfg["data_dir"],
        int(cfg["img_size"]),
        int(cfg["batch_size"]),
        int(cfg["num_workers"]),
        augment_train=False,
    )

    latent_dim = int(cfg["vae"]["latent_dim"])
    vae = VAE(latent_dim=latent_dim).to(device)

    raw_lr = cfg["vae"]["lr"]
    lr = float(raw_lr)
    print("Loaded VAE lr type/value:", type(raw_lr), raw_lr)
    print("Using VAE lr:", lr)

    opt = torch.optim.Adam(vae.parameters(), lr=lr)

    epochs = int(cfg["vae"]["epochs"])
    for epoch in range(1, epochs + 1):
        vae.train()
        pbar = tqdm(train_loader, desc=f"VAE epoch {epoch}/{epochs}")

        for x, _y in pbar:
            x = x.to(device)

            opt.zero_grad(set_to_none=True)
            x_hat, mu, lv = vae(x)
            loss, recon, kld = vae_loss(x, x_hat, mu, lv)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.item()), recon=float(recon.item()), kld=float(kld.item()))

        torch.save({"vae": vae.state_dict()}, "vae_last.pt")


if __name__ == "__main__":
    main()

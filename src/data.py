from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def build_transforms(img_size: int, train: bool):
    """
    train=True  -> augmentation pipeline (for classifier training)
    train=False -> clean pipeline (for eval, and for VAE training)
    """
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),  # [0,1]
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # [0,1]
        ])


def build_loaders(
    data_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    augment_train: bool = True,
):
    """
    augment_train=True  -> training loader uses augmentation (classifier)
    augment_train=False -> training loader uses clean transforms (VAE)
    """
    root: Path = Path(data_dir)

    train_tf = build_transforms(img_size, train=augment_train)
    eval_tf = build_transforms(img_size, train=False)

    train_ds = datasets.ImageFolder(str(root / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(root / "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(str(root / "test"), transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_ds.class_to_idx

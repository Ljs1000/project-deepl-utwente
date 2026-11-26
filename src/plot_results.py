from pathlib import Path
import csv
import matplotlib.pyplot as plt


def read_csv(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def floats(rows, key):
    return [float(r[key]) for r in rows]


def plot_line(x, y, xlabel, ylabel, title, out_path: Path, label=None):
    plt.figure()
    if label is None:
        plt.plot(x, y, marker="o")
    else:
        plt.plot(x, y, marker="o", label=label)
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path.resolve())


def main():
    out_dir = Path("outputs")
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- FGSM attack results ---
    fgsm_csv = out_dir / "fgsm_attack.csv"
    if fgsm_csv.exists():
        fgsm = read_csv(fgsm_csv)
        eps = floats(fgsm, "epsilon")
        acc = floats(fgsm, "adv_acc")
        plot_line(
            eps, acc,
            "Epsilon", "Adversarial accuracy",
            "Classifier robustness under FGSM",
            plot_dir / "fgsm_accuracy_vs_epsilon.png"
        )
    else:
        print("Missing:", fgsm_csv, "(run python -m src.eval_attack)")

    # --- PGD attack results ---
    pgd_csv = out_dir / "pgd_attack.csv"
    if pgd_csv.exists():
        pgd = read_csv(pgd_csv)
        eps = floats(pgd, "epsilon")
        acc = floats(pgd, "pgd_acc")
        plot_line(
            eps, acc,
            "Epsilon", "Adversarial accuracy",
            "Classifier robustness under PGD",
            plot_dir / "pgd_accuracy_vs_epsilon.png"
        )
    else:
        print("Missing:", pgd_csv, "(run python -m src.eval_pgd_attack)")

    # --- FGSM detector results ---
    det_fgsm_csv = out_dir / "detector_laplacian.csv"
    if det_fgsm_csv.exists():
        det = read_csv(det_fgsm_csv)
        eps = floats(det, "epsilon")
        auc = floats(det, "auc")
        plot_line(
            eps, auc,
            "Epsilon", "AUC",
            "VAE-Laplacian detector (FGSM)",
            plot_dir / "detector_auc_fgsm.png"
        )
    else:
        print("Missing:", det_fgsm_csv, "(run python -m src.eval_detector)")

    # --- PGD detector results ---
    det_pgd_csv = out_dir / "detector_laplacian_pgd.csv"
    if det_pgd_csv.exists():
        det = read_csv(det_pgd_csv)
        eps = floats(det, "epsilon")
        auc = floats(det, "auc")
        plot_line(
            eps, auc,
            "Epsilon", "AUC",
            "VAE-Laplacian detector (PGD)",
            plot_dir / "detector_auc_pgd.png"
        )
    else:
        print("Missing:", det_pgd_csv, "(run python -m src.eval_detector_pgd)")


if __name__ == "__main__":
    main()

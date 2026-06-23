"""
Wstep do sztucznej inteligencji - Lab 3
Temat: Support Vector Machines / SVC
Zbior danych: Breast Cancer Wisconsin

Jak uruchomic:
1. pip install numpy matplotlib scikit-learn
2. python svm_breast_cancer_lab3.py

Skrypt generuje:
- tabele z wynikami accuracy,
- macierze konfuzji,
- wykresy 2D dla rzeczywistych i przewidzianych klas,
- pliki PNG/CSV w katalogu svm_outputs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
TEST_SIZE = 0.3
C_VALUES = [0.01, 0.1, 1, 10, 100]
KERNELS = ["linear", "rbf"]
OUT_DIR = Path("svm_outputs")
OUT_DIR.mkdir(exist_ok=True)


def train_and_evaluate(X_train, X_test, y_train, y_test, kernel, C):
    """Trenuje SVC i zwraca model, predykcje oraz accuracy."""
    model = SVC(kernel=kernel, C=C, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, y_pred, acc


def save_confusion_matrix(y_true, y_pred, target_names, title, filename):
    """Zapisuje macierz konfuzji do pliku PNG."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200)
    plt.close(fig)
    return cm


def save_scatter_plot(X, labels, feature_names, feature_idx, title, filename):
    """Zapisuje wykres punktowy 2D dla dwoch wybranych cech."""
    f1, f2 = feature_idx
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(X[:, f1], X[:, f2], c=labels, alpha=0.75, edgecolors="k", linewidths=0.3)
    ax.set_xlabel(feature_names[f1])
    ax.set_ylabel(feature_names[f2])
    ax.set_title(title)
    legend = ax.legend(*scatter.legend_elements(), title="Klasa")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200)
    plt.close(fig)


def save_accuracy_plot(results):
    """Zapisuje wykres wplywu C i kernela na accuracy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for preprocessing_name in ["bez standaryzacji", "ze standaryzacja"]:
        for kernel in KERNELS:
            subset = [r for r in results if r["preprocessing"] == preprocessing_name and r["kernel"] == kernel]
            xs = [r["C"] for r in subset]
            ys = [r["accuracy"] for r in subset]
            ax.plot(xs, ys, marker="o", label=f"{preprocessing_name}, {kernel}")
    ax.set_xscale("log")
    ax.set_xlabel("C")
    ax.set_ylabel("accuracy")
    ax.set_title("Wplyw parametru C, kernela i standaryzacji na accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_plot.png", dpi=200)
    plt.close(fig)


def main():
    # 1. Wczytanie danych
    data = load_breast_cancer()
    X = data.data
    y = data.target
    target_names = data.target_names
    feature_names = data.feature_names

    print("Rozmiar macierzy cech X:", X.shape)
    print("Liczba klas:", len(np.unique(y)))
    print("Nazwy klas:", target_names)

    # 2. Podzial 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Rozmiar zbioru uczacego:", X_train.shape)
    print("Rozmiar zbioru testowego:", X_test.shape)

    # 3. Standaryzacja - fit tylko na danych uczacych
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Test roznych konfiguracji C i kernel, ze standaryzacja oraz bez niej
    results = []
    for preprocessing_name, Xt, Xv in [
        ("bez standaryzacji", X_train, X_test),
        ("ze standaryzacja", X_train_scaled, X_test_scaled),
    ]:
        for kernel in KERNELS:
            for C in C_VALUES:
                _, y_pred, acc = train_and_evaluate(Xt, Xv, y_train, y_test, kernel, C)
                results.append(
                    {
                        "preprocessing": preprocessing_name,
                        "kernel": kernel,
                        "C": C,
                        "accuracy": acc,
                    }
                )

    # Zapis tabeli CSV
    csv_path = OUT_DIR / "accuracy_results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("preprocessing,kernel,C,accuracy\n")
        for row in results:
            f.write(f"{row['preprocessing']},{row['kernel']},{row['C']},{row['accuracy']:.6f}\n")

    save_accuracy_plot(results)

    print("\nWyniki accuracy:")
    for row in results:
        print(f"{row['preprocessing']:18s} | kernel={row['kernel']:6s} | C={row['C']:6g} | accuracy={row['accuracy']:.4f}")

    # 5. Wybor najlepszych konfiguracji dla linear i rbf na danych standaryzowanych
    scaled_results = [r for r in results if r["preprocessing"] == "ze standaryzacja"]
    best_by_kernel = {}
    for kernel in KERNELS:
        kernel_results = [r for r in scaled_results if r["kernel"] == kernel]
        best_by_kernel[kernel] = max(kernel_results, key=lambda r: r["accuracy"])

    print("\nNajlepsze konfiguracje po standaryzacji:")
    for kernel, row in best_by_kernel.items():
        print(f"kernel={kernel}, C={row['C']}, accuracy={row['accuracy']:.4f}")

    # 6. Macierze konfuzji i wykresy 2D dla najlepszych modeli
    # Wybrane cechy do wizualizacji 2D: mean radius i mean texture
    feature_idx = (0, 1)

    for kernel, best in best_by_kernel.items():
        C = best["C"]
        model, y_pred, acc = train_and_evaluate(
            X_train_scaled, X_test_scaled, y_train, y_test, kernel, C
        )

        cm = save_confusion_matrix(
            y_test,
            y_pred,
            target_names,
            f"Macierz konfuzji - SVC {kernel}, C={C}, acc={acc:.4f}",
            f"confusion_matrix_{kernel}.png",
        )
        print(f"\nMacierz konfuzji dla kernel={kernel}, C={C}:")
        print(cm)

        save_scatter_plot(
            X_test,
            y_test,
            feature_names,
            feature_idx,
            f"Rzeczywisty podzial klas - {kernel}, C={C}",
            f"scatter_true_{kernel}.png",
        )

        save_scatter_plot(
            X_test,
            y_pred,
            feature_names,
            feature_idx,
            f"Podzial klas przewidziany przez model - {kernel}, C={C}",
            f"scatter_pred_{kernel}.png",
        )

    print(f"\nPliki wynikowe zapisano w katalogu: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

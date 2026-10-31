
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "confidence_recovery"
OUTPUT_DIR = RESULTS_DIR / "plots"
INDIVIDUAL_DIR = RESULTS_DIR / "plots_individual_pdf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)

P_ERRORS_LIST = [0.5]
VARIANTS = ["PANDA-S", "PANDA-D", "PANDA-F"]
DATASET_ORDER = ["Bupa", "Ionosphere", "Breast", "Pima", "TicTacToe", "Occupancy", "Skin_NonSkin", "Iris", "Wine", "Segmentation", "Western"]

sns.set_theme(style="white")

for p_errors in P_ERRORS_LIST:
    error_tag = f"{p_errors:g}"
    CSV_PATH = RESULTS_DIR / f"confidence_recovery_summary_{error_tag}_es.csv"

    if not CSV_PATH.exists():
        print(f"No se encuentra: {CSV_PATH}")
        continue

    df = pd.read_csv(CSV_PATH)

    for model in VARIANTS:
        model_df = df[df["variant"] == model]
        model_datasets = [d for d in DATASET_ORDER if (model_df["dataset"] == d).any()]

        if not model_datasets:
            print(f"Sin resultados para {model}, p_errors={error_tag}")
            continue

        # Carpeta para los PDF individuales de este modelo y p_errors.
        individual_model_dir = INDIVIDUAL_DIR / f"confidence_recovery_{model.replace('-', '')}_{error_tag}"
        individual_model_dir.mkdir(parents=True, exist_ok=True)

        ncols = 4
        nrows = math.ceil(len(model_datasets) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.2 * nrows), squeeze=False)
        fig.subplots_adjust(left=0.06, right=0.91, top=0.90, bottom=0.09, hspace=0.1, wspace=0.1)

        for i, ax in enumerate(axes.flat):
            if i >= len(model_datasets):
                ax.set_visible(False)
                continue

            dataset = model_datasets[i]
            subset = model_df[model_df["dataset"] == dataset]
            matrix = subset.pivot(index="cluster", columns="annotator", values="confidence")
            matrix = matrix.reindex(
                index=sorted(matrix.index, key=lambda x: int(str(x).split("_")[-1])),
                columns=sorted(matrix.columns, key=lambda x: int(str(x).split("_")[-1]))
            )
            matrix.index = [f"C{j}" for j in range(1, len(matrix.index) + 1)]
            matrix.columns = [f"A{j}" for j in range(1, len(matrix.columns) + 1)]

            row, col = divmod(i, ncols)
            show_y, show_x = col == 0, row == nrows - 1

            sns.heatmap(
                matrix, ax=ax, cmap="plasma", vmin=0, vmax=1,
                annot=True, fmt=".2f", square=True, cbar=False,
                annot_kws={"size": 7},
                xticklabels=show_x, yticklabels=show_y
            )
            ax.set_title(dataset, fontsize=10, pad=5)
            ax.set_xlabel("Anotador" if show_x else "", fontsize=9, labelpad=2)
            ax.set_ylabel("Clúster" if show_y else "", fontsize=9, labelpad=3)
            ax.tick_params(axis="both", labelsize=8, length=0)

            # ---- Guardar matriz individual en PDF sin elementos adicionales ----
            fig_ind, ax_ind = plt.subplots(figsize=(5, 5))
            sns.heatmap(
                matrix, ax=ax_ind, cmap="plasma", vmin=0, vmax=1,
                annot=True, fmt=".2f", square=True, cbar=False,
                annot_kws={"size": 9},
                xticklabels=False, yticklabels=False
            )
            ax_ind.set_axis_off()
            fig_ind.savefig(
                individual_model_dir / f"matrix_{dataset}_{model}.pdf",
                bbox_inches="tight",
                pad_inches=0
            )
            plt.close(fig_ind)

        # Barra de color compartida, separada de las matrices.
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar_ax = fig.add_axes([0.935, 0.20, 0.015, 0.60])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Confianza", fontsize=9, labelpad=7)
        cbar.ax.tick_params(labelsize=8)

        fig.suptitle(f"Recuperación de confianza — {model} — p_errors={error_tag}", fontsize=13, y=0.96)
        output_path = OUTPUT_DIR / f"confidence_recovery_{model.replace('-', '')}_{error_tag}_auc.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Imagen guardada: {output_path}")
        print(f"PDF individuales guardados en: {individual_model_dir}")

print("¡Gráficas generales y PDF individuales generados!")

import matplotlib.pyplot as plt
import seaborn as sns
from os import path, makedirs

current_dir = path.dirname(path.abspath(__file__))
output_path = path.join(current_dir, "../output")


def save_show_heatmap_as_image(
    confusion_matrix, folder="default", filename="heatmap.png", title="Confusion matrix"
):
    heatmap_path = path.join(output_path, "heatmaps", folder)
    makedirs(heatmap_path, exist_ok=True)

    plt.figure()
    sns.heatmap(confusion_matrix, annot=True, fmt="d")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    file_path = path.join(heatmap_path, filename)

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Heatmap guardado en: {file_path}")


if __name__ == "__main__":
    confusion_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    save_show_heatmap_as_image(
        confusion_matrix,
        folder="default",
        filename="heatmap.png",
        title="Confusion matrix",
    )

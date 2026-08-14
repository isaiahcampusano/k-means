"""Compress RGB images with K-Means color quantization.

Every pixel is represented by a three-dimensional ``[red, green, blue]``
vector. K-Means learns a small palette of representative vectors, then
replaces every pixel with the closest palette color.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)


def load_image(path: Path | str) -> np.ndarray:
    """Load an image from ``path`` as an RGB NumPy array."""

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def compress_image(
    pixel_data: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce an RGB image to at most ``k`` representative colors.

    Args:
        pixel_data: Image data with shape ``(height, width, 3)``.
        k: Number of color clusters to learn.

    Returns:
        The compressed image, centroid colors, cluster label per pixel, and
        flattened original RGB vectors.

    Raises:
        ValueError: If the input is not an RGB image or ``k`` is invalid.
    """

    if pixel_data.ndim != 3 or pixel_data.shape[2] != 3:
        raise ValueError("pixel_data must have shape (height, width, 3)")
    if k < 1:
        raise ValueError("k must be at least 1")

    height, width, channels = pixel_data.shape
    vectors = pixel_data.reshape(-1, channels)
    if k > len(vectors):
        raise ValueError("k cannot exceed the number of pixels")

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(vectors)
    centers = np.clip(np.rint(model.cluster_centers_), 0, 255).astype(np.uint8)
    compressed = centers[labels].reshape(height, width, channels)
    return compressed, centers, labels, vectors


def save_palette(centers: np.ndarray, path: Path | str) -> None:
    """Save centroid colors as a labeled swatch grid."""

    palette_path = Path(path)
    palette_path.parent.mkdir(parents=True, exist_ok=True)
    count = len(centers)
    columns = min(count, 8)
    rows = int(np.ceil(count / columns))
    swatch_size = 80

    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.25, rows * 1.25))
    flat_axes = np.atleast_1d(axes).ravel()
    for index, axis in enumerate(flat_axes):
        axis.set_axis_off()
        if index < count:
            color = centers[index]
            axis.imshow(np.full((swatch_size, swatch_size, 3), color, dtype=np.uint8))
            axis.set_title(
                f"{color[0]}, {color[1]}, {color[2]}",
                fontsize=8,
                color="#f4efe7",
                pad=6,
            )
    figure.patch.set_facecolor("#151914")
    figure.tight_layout(pad=1.2)
    figure.savefig(palette_path, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_rgb_scatter(
    vectors: np.ndarray,
    centers: np.ndarray,
    path: Path | str,
    sample_size: int = 5_000,
) -> None:
    """Plot sampled pixel vectors and learned centroids in RGB space."""

    scatter_path = Path(path)
    scatter_path.parent.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(42)
    indices = generator.choice(
        len(vectors), size=min(sample_size, len(vectors)), replace=False
    )
    sample = vectors[indices]

    figure = plt.figure(figsize=(8, 8), facecolor="#151914")
    axis = figure.add_subplot(111, projection="3d", facecolor="#151914")
    axis.scatter(
        sample[:, 0],
        sample[:, 1],
        sample[:, 2],
        c=sample / 255.0,
        s=5,
        alpha=0.35,
        linewidths=0,
    )
    axis.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c=centers / 255.0,
        edgecolors="#ffffff",
        linewidths=1.5,
        marker="^",
        s=140,
        label="K-Means centroids",
    )
    axis.set(xlabel="Red", ylabel="Green", zlabel="Blue")
    axis.xaxis.label.set_color("#c7cbbf")
    axis.yaxis.label.set_color("#c7cbbf")
    axis.zaxis.label.set_color("#c7cbbf")
    axis.set_title(f"Pixel vectors in RGB space · k={len(centers)}", color="#f4efe7")
    axis.tick_params(colors="#c7cbbf")
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((0.10, 0.12, 0.10, 1.0))
    axis.legend(facecolor="#242a22", labelcolor="#f4efe7", edgecolor="#65705d")
    figure.tight_layout()
    figure.savefig(scatter_path, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


def output_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    """Build all result paths from the input filename."""

    stem = input_path.stem
    return (
        output_dir / f"{stem}_compressed.jpg",
        output_dir / f"{stem}_palette.png",
        output_dir / f"{stem}_rgb_scatter.png",
    )


def run(input_path: Path, k: int, output_dir: Path) -> tuple[Path, Path, Path]:
    """Run the complete compression pipeline and save its artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    compressed_path, palette_path, scatter_path = output_paths(input_path, output_dir)
    pixels = load_image(input_path)
    LOGGER.info("Loaded %s (%d × %d pixels)", input_path, pixels.shape[1], pixels.shape[0])
    compressed, centers, _labels, vectors = compress_image(pixels, k)
    Image.fromarray(compressed).save(compressed_path, quality=75, optimize=True)
    save_palette(centers, palette_path)
    plot_rgb_scatter(vectors, centers, scatter_path)
    LOGGER.info("Created %s, %s, and %s", compressed_path, palette_path, scatter_path)
    return compressed_path, palette_path, scatter_path


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description="Compress an image using K-Means color quantization."
    )
    parser.add_argument("--input", type=Path, required=True, help="Source image path")
    parser.add_argument("--k", type=int, default=16, help="Number of colors (default: 16)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated files (default: outputs/)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line application."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    run(args.input, args.k, args.output_dir)


if __name__ == "__main__":
    main()

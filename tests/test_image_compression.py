"""Tests for K-Means image compression."""

import numpy as np

from src.image_compression import compress_image


def test_compress_image_limits_palette_and_preserves_shape() -> None:
    """Compression returns the original shape with no more than k colors."""

    generator = np.random.default_rng(42)
    pixels = generator.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

    compressed, centers, labels, vectors = compress_image(pixels, k=4)

    assert compressed.shape == pixels.shape
    assert len(np.unique(compressed.reshape(-1, 3), axis=0)) <= 4
    assert centers.shape == (4, 3)
    assert labels.shape == (32 * 32,)
    assert vectors.shape == (32 * 32, 3)


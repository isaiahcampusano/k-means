# Image compression with K-Means

This project treats every image pixel as a vector—`[red, green, blue]`—and
uses K-Means clustering to learn a compact palette. Each original pixel is
then replaced by its nearest learned color. The included example uses Isaiah's
dog photo and a 16-color palette.

![Uploading image.png…]()


## Try the live explainer

The published project includes an interactive before/after comparison, the
learned palette, and a 3D view of the pixel vectors:

**https://isaiahcampusano.github.io/k-means/**

## Run it locally

Create a virtual environment, install the dependencies, and run:

```bash
python -m src.image_compression --input IMG_0959.jpeg --k 16 --output-dir outputs
```

The command derives these filenames from the input:

- `outputs/IMG_0959_compressed.jpg` — the quantized image
- `outputs/IMG_0959_palette.png` — the learned centroid colors
- `outputs/IMG_0959_rgb_scatter.png` — pixel vectors plotted in RGB space

Use a smaller `--k` for fewer colors and stronger compression, or a larger
value for more detail.

## Test it

```bash
pytest
```

The test builds a synthetic RGB image, compresses it to four clusters, and
checks that its shape is preserved and its palette contains no more than four
colors.

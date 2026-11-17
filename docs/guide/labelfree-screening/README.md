# Label-free screening tutorial: source files

This folder is self-contained: the tutorial page, the 2-page whitepaper PDF
source, the figures, the configs the tutorial teaches, and the scripts that
regenerate everything.

| Path | Contents |
| --- | --- |
| `labelfree-screening.md` | The tutorial page (rendered on the docs site). |
| `whitepaper.tex` | 2-page print layout of the same content (boxed failure modes, Biohub footer). |
| `whitepaper.pdf` | The built whitepaper. |
| `make_pdf.sh` | Builds `whitepaper.pdf` with pdflatex. |
| `figures/` | Tutorial figures: PDF (whitepaper) and PNG (docs site). |
| `figures/make_figures.py` | Draws the data figures from the crops in `figures/data/`. |
| `figures/data/` | 8-bit image crops behind the data figures, each scaled to the contrast its panel shows. |
| `figures/koehler/` | Draws the Koehler alignment figure from a parametric cartoon microscope (`scope_model.py`, `make_figure.py`). |
| `figures/requirements.txt` | Pinned packages for both figure scripts, compiled from `requirements.in`. |
| `configs/phase.yml` | Baseline 2D phase reconstruction config. |
| `configs/phase_autofocus.yml` | Baseline plus `z_focus_offset: auto`. |
| `configs/phase_autofocus_tilt.yml` | Autofocus plus warm-started illumination tilt fitting. |
| `brand/` | CZ Biohub logo for the whitepaper footer. |

## Rebuilding

You need [uv](https://docs.astral.sh/uv/), and for the PDF a full TeX Live or
MacTeX install (`make_pdf.sh` lists the LaTeX packages). Nothing outside this
folder is read.

```sh
cd figures
uv run --no-project --python 3.12 --with-requirements requirements.txt python make_figures.py
uv run --no-project --python 3.12 --with-requirements requirements.txt python koehler/make_figure.py
cd .. && ./make_pdf.sh
```

- The first run downloads cadquery and vtk (several hundred MB). The Koehler
  figure takes one to two minutes, because it renders a 3D model; on a
  headless Linux machine vtk needs an OpenGL context (e.g. `xvfb-run`).
- With the pinned packages the figures come out byte-for-byte identical to
  the committed ones on the machine they were made on (macOS, Apple
  silicon). Other platforms may differ in rendering details.
- `whitepaper.pdf` is built without timestamps, so rebuilding with the same
  TeX install reproduces it exactly. The date in its footer is set by hand in
  `whitepaper.tex`.

## Where the crops come from

The crops in `figures/data/` are cut from one raw position of the OPS atlas
store (`A/1/029029`, well center) and from `waveorder reconstruct` outputs
for that position and for `A/1/002029` (north edge), using the configs in
`configs/` and variants of them with the one setting each panel names
changed. `python figures/make_figures.py --from-hpc` rebuilds the crops from
those stores; it needs Biohub HPC access and iohub, and its docstring lists
the reconstructions it expects.

The committed crops were recovered from the figure PDFs of an earlier
revision of this tutorial, at 161 to 248 pixels across. Rebuilding them with
`--from-hpc` gives 256-pixel crops cut from the full-resolution data.

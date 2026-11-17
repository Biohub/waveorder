---
orphan: true
---

# Label-free screening tutorial: source files

This folder is self-contained: the tutorial page, the 2-page whitepaper PDF
source, the figures, the configs the tutorial teaches, and the scripts that
regenerate everything.

| Path | Contents |
| --- | --- |
| `labelfree-screening.md` | The tutorial page (rendered on the docs site). |
| `whitepaper.tex` | 2-page print layout of the same content (boxed failure modes, Biohub footer). |
| `make_pdf.sh` | Builds `whitepaper.pdf` with pdflatex (base texlive packages only). |
| `figures/` | Tutorial figures: vector PDF (whitepaper) and PNG (docs site). |
| `figures/make_figures.py` | Regenerates the data figures from the crops in `figures/data/`; runs anywhere. |
| `figures/data/` | 8-bit, display-scaled image crops behind the figures. `make_figures.py --from-hpc` rebuilds them from the OPS atlas data and reconstruction outputs (requires Biohub HPC access). |
| `figures/koehler/` | Regenerates the Koehler alignment figure from a parametric cartoon microscope (`scope_model.py`, `make_figure.py`; requires cadquery). |
| `configs/phase.yml` | Baseline 2D phase reconstruction config. |
| `configs/phase_autofocus.yml` | Baseline plus `z_focus_offset: auto`. |
| `configs/phase_autofocus_tilt.yml` | Autofocus plus warm-started illumination tilt fitting. |
| `brand/` | CZ Biohub logo for the whitepaper footer. |

To update a figure, edit and rerun its script, then rebuild the PDF:

```sh
cd figures
uv run --no-project --python 3.12 --with matplotlib==3.10.8 --with pillow==13.0.0 --with numpy==2.4.2 \
    python make_figures.py
uv run --no-project --python 3.12 --with cadquery==2.8.0 --with matplotlib==3.10.8 --with pillow==13.0.0 \
    python koehler/make_figure.py
cd .. && ./make_pdf.sh
```

The pinned versions are the ones the committed figures were made with. The
figure scripts need no data outside this folder.

All reconstructions behind the crops are plain `waveorder reconstruct` calls
with the configs in `configs/` on positions `A/1/029029` (well center) and
`A/1/002029` (north edge) of the OPS atlas store.

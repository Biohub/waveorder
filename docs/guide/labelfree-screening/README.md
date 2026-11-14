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
| `figures/make_figures.py` | Regenerates the figures. Requires Biohub HPC access to the OPS atlas data and reconstruction outputs; see the module docstring for the analysis directory. |
| `configs/phase.yml` | Baseline 2D phase reconstruction config. |
| `configs/phase_autofocus.yml` | Baseline plus `z_focus_offset: auto`. |
| `configs/phase_autofocus_tilt.yml` | Autofocus plus warm-started illumination tilt fitting. |
| `brand/` | CZ Biohub logo for the whitepaper footer. |

To update a figure: edit `figures/make_figures.py`, run it on the Biohub HPC
(it writes into the analysis directory), and copy the outputs here. All
reconstructions behind the figures are plain `waveorder reconstruct` calls
with the configs in `configs/` on positions `A/1/029029` (well center) and
`A/1/002029` (north edge) of the OPS atlas store.

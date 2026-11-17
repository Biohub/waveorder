#!/bin/bash
# Build the 2-page whitepaper PDF from whitepaper.tex.
#
# Needs pdflatex with the LaTeX packages extsizes, geometry, helvet, textcomp,
# graphicx, xcolor, fancyvrb, booktabs, caption, enumitem, fancyhdr and
# hyperref (all in a full TeX Live or MacTeX install).
set -euo pipefail
cd "$(dirname "$0")"
trap 'rm -f whitepaper.aux whitepaper.out' EXIT
for pass in 1 2; do
    if ! pdflatex -interaction=nonstopmode -halt-on-error whitepaper.tex >/dev/null; then
        echo "pdflatex failed; last lines of whitepaper.log:" >&2
        tail -n 25 whitepaper.log >&2
        exit 1
    fi
done
rm -f whitepaper.log
# The layout is designed for exactly two pages; report the count if poppler is installed.
if command -v pdfinfo >/dev/null; then
    pdfinfo whitepaper.pdf | grep Pages
fi

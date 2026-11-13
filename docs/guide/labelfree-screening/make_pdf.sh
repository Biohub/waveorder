#!/bin/bash
# Render the 2-page whitepaper PDF from whitepaper.tex.
# Needs only base texlive packages (pdflatex, fancyvrb, booktabs, caption).
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode whitepaper.tex >/dev/null
pdflatex -interaction=nonstopmode whitepaper.tex >/dev/null
rm -f whitepaper.aux whitepaper.log whitepaper.out
pdfinfo whitepaper.pdf | grep Pages

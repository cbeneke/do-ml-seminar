#!/usr/bin/env bash

set -e
trap 'rm -f main.{aux,bbl,blg,log,out,pdf}' EXIT

pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex # Run twice to resolve references and citations

mv main.pdf ../paper.pdf
#!/usr/bin/env bash

set -e
trap 'rm -f main.{aux,bbl,blg,log,out,pdf}' EXIT

pdflatex main.tex
bibtex main
pdflatex main.tex
mv main.pdf ../paper.pdf
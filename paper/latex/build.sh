#!/usr/bin/env bash

pdflatex main.tex
pdflatex main.tex # Run twice to resolve references
mv main.pdf ../paper.pdf
rm main.{aux,bbl,blg,log,out}

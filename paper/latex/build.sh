#!/usr/bin/env bash

pdflatex main.tex
mv main.pdf ../paper.pdf
rm main.{aux,log,out}

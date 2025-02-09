#!/usr/bin/env bash

set -e
trap 'rm -f main.{aux,log,nav,out,pdf,snm,toc}' EXIT

pdflatex main.tex

mv main.pdf ../presentation.pdf

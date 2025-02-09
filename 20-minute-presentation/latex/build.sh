#!/usr/bin/env bash

set -e
trap 'rm -f main.{aux,log,nav,out,pdf,snm,toc}' EXIT

pdflatex main.tex
pdflatex main.tex # Build twice for the TOC and Nav

mv main.pdf ../presentation.pdf

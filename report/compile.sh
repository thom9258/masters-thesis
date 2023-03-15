#!/bin/sh

pdflatex main.tex && makeglossaries main && pdflatex main.tex

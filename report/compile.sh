#!/bin/sh

biber main && \
    #pdflatex main.tex && \
    biber main && \
    makeglossaries main && \
    pdflatex main.tex

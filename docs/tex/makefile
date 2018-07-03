#!/bin/sh

pdflatex fitr_documentation
bibtex fitr_documentation
pdflatex fitr_documentation
pdflatex fitr_documentation
pdflatex fitr_documentation
pdflatex fitr_documentation


find . -type f -name '*.aux' -delete
find . -type f -name '*.bbl' -delete
find . -type f -name '*.blg' -delete
find . -type f -name '*.bcf' -delete
find . -type f -name '*.-blx.bib' -delete
find . -type f -name '*.log' -delete
find . -type f -name '*.out' -delete
find . -type f -name '*.run.xml' -delete
find . -type f -name '*.toc' -delete
find . -type f -name '*.lof' -delete
find . -type f -name '*.lot' -delete
find . -type f -name '*.lol' -delete
find . -type f -name '*.idx' -delete
find . -type f -name '*.ilg' -delete
find . -type f -name '*.ind' -delete
find . -type f -name '*.tdo' -delete
find . -type f -name '*.dvi' -delete
find . -type f -name '*.fdb_latexmk' -delete
find . -type f -name '*.fls' -delete
find . -type f -name 'comment.cut' -delete

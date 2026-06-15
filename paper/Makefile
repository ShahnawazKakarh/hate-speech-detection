# Build the paper:
#   make            -> main.pdf
#   make clean      -> remove aux files
#   make distclean  -> + remove main.pdf

TEX   := main
LATEX := pdflatex -interaction=nonstopmode -halt-on-error
BIB   := bibtex

$(TEX).pdf: $(TEX).tex references.bib tables/results.tex
	$(LATEX) $(TEX)
	$(BIB) $(TEX) || true
	$(LATEX) $(TEX)
	$(LATEX) $(TEX)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(TEX).pdf

.PHONY: clean distclean

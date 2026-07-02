# Build the paper PDF via WeasyPrint (no LaTeX required).
#
#   make               -> main.pdf  (WeasyPrint from cover.html + main.md + theme.css)
#   make latex         -> main.pdf  (fall-back path: pdflatex + bibtex on main.tex)
#   make clean         -> remove build artefacts
#   make distclean     -> + remove main.pdf

.PHONY: all clean distclean latex

all: main.pdf

main.pdf: cover.html main.md theme.css
	@echo "→ WeasyPrint build"
	python build_pdf.py

# LaTeX fall-back (requires MacTeX / pdflatex on PATH)
LATEX  := pdflatex -interaction=nonstopmode -halt-on-error
BIB    := bibtex
TEX    := main

latex:
	@echo "→ pdflatex build"
	$(LATEX) $(TEX)
	$(BIB) $(TEX) || true
	$(LATEX) $(TEX)
	$(LATEX) $(TEX)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz _build_debug.html

distclean: clean
	rm -f main.pdf

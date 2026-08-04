PYTHON ?= python3
PDFLATEX ?= pdflatex
SOURCE_DATE_EPOCH ?= 1785801600

export SOURCE_DATE_EPOCH
export FORCE_SOURCE_DATE := 1

PAPER_DIR := paper
PAPER_NAME := cortex_quake_bc
ARCHIVE := $(PAPER_DIR)/releases/cortex-quake-bc-arxiv.tar.gz

.PHONY: paper figures arxiv clean-paper

paper: figures
	cd $(PAPER_DIR) && $(PDFLATEX) -interaction=nonstopmode -halt-on-error $(PAPER_NAME).tex
	cd $(PAPER_DIR) && $(PDFLATEX) -interaction=nonstopmode -halt-on-error $(PAPER_NAME).tex

figures:
	MPLBACKEND=Agg $(PYTHON) scripts/make_architecture_figure.py
	MPLBACKEND=Agg $(PYTHON) scripts/make_inference_latency_figure.py
	MPLBACKEND=Agg $(PYTHON) scripts/make_waypoint_survival_figure.py

arxiv: paper
	tar --sort=name --mtime='UTC 2026-08-04' --owner=0 --group=0 --numeric-owner \
		-czf $(ARCHIVE) -C $(PAPER_DIR) $(PAPER_NAME).tex \
		figures/architecture.pdf figures/inference_latency_rtx5080.pdf \
		figures/waypoint_survival.pdf
	sha256sum $(ARCHIVE)

clean-paper:
	$(RM) $(PAPER_DIR)/$(PAPER_NAME).aux $(PAPER_DIR)/$(PAPER_NAME).log \
		$(PAPER_DIR)/$(PAPER_NAME).out $(PAPER_DIR)/$(PAPER_NAME).pdf \
		$(PAPER_DIR)/figures/architecture.pdf $(PAPER_DIR)/figures/architecture.png \
		$(PAPER_DIR)/figures/inference_latency_rtx5080.pdf \
		$(PAPER_DIR)/figures/inference_latency_rtx5080.png \
		$(PAPER_DIR)/figures/waypoint_survival.pdf \
		$(PAPER_DIR)/figures/waypoint_survival.png $(ARCHIVE)

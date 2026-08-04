# arXiv submission bundle

Install the optional plotting dependencies and generate
**cortex-quake-bc-arxiv.tar.gz** from the repository root with:

    pip install -e '.[paper]'
    make arxiv

The generated manuscript, figures, and archive are ignored by Git. Upload the
archive as the arXiv source package; it contains only:

- cortex_quake_bc.tex
- figures/architecture.pdf
- figures/inference_latency_rtx5080.pdf
- figures/waypoint_survival.pdf

The package deliberately excludes Python source, generated raster figures,
the compiled manuscript, and build logs. The paper links to the public code
repository instead.

`make arxiv` compiles the manuscript twice before packaging and prints the
archive's SHA-256 checksum.

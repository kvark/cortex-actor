# arXiv submission bundle

Generate **cortex-quake-bc-arxiv.tar.gz** from the repository root with:

    tar --sort=name --mtime='UTC 2026-08-04' --owner=0 --group=0 \
      --numeric-owner -czf paper/releases/cortex-quake-bc-arxiv.tar.gz \
      -C paper cortex_quake_bc.tex figures/architecture.pdf \
      figures/inference_latency_rtx5080.pdf figures/waypoint_survival.pdf

The generated archive is ignored by Git. Upload it as the arXiv source
package; it contains only:

- cortex_quake_bc.tex
- figures/architecture.pdf
- figures/inference_latency_rtx5080.pdf
- figures/waypoint_survival.pdf

The package deliberately excludes Python source, generated raster figures,
the compiled manuscript, and build logs. The paper links to the public code
repository instead.

SHA-256: **5d45077d4ad1b1dde3c60770dcbd4beee275bc988d3fbba81eabc7771b123fc0**

It was verified by extracting into an empty directory and compiling
cortex_quake_bc.tex with pdfLaTeX.

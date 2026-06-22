0.61.0 (June 22, 2026)
======================
Adds a Sphinx documentation site and hardens the test suite; the DSC and DCE
processing pipelines are unchanged from the prior series.

* DOC: Add a Sphinx documentation site (ReadTheDocs-ready) with an auto-generated API reference, a ``sphinx-argparse`` command-line reference, and a rendered changelog
* DOC: Fix reStructuredText formatting across package docstrings (math literals, literal blocks, escapes) for a clean documentation build
* FIX: Make the workflow graph-regression baselines hardware-independent so the suite passes on CPU-only CI runners (``use_gpu`` is no longer captured in the graph signature)
* TST: Drop redundant graph-regression matrix variants subsumed by the all-underlays cases (22 to 13)
* MNT: Add the ``doc`` optional-dependency group and a documentation build job to CI

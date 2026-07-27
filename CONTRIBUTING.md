# Contributing to PMRIPrep

PMRIPrep follows the [NiPreps contribution guidelines](https://www.nipreps.org/community/CONTRIBUTING/).
Please open or join an issue before starting a substantial change.

Install a development checkout with:

```bash
python -m pip install -e ".[dev]"
```

Run tests and static checks in the project development environment:

```bash
conda run -n pmriprep-env pytest tests/unit/ -q
conda run -n pmriprep-env ruff check pmriprep/ tests/
conda run -n pmriprep-env mypy pmriprep/ --ignore-missing-imports
```

Changes to quantitative methods, constants, or QC thresholds must include a
scientific rationale and focused validation. Keep pull requests reviewable and
update user documentation when behavior or configuration changes.

By contributing, you agree that your contribution is licensed under the
project's Apache-2.0 license and to follow the [PMRIPrep code of conduct](CODE_OF_CONDUCT.md).

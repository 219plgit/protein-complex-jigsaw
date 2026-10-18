# Applying the Protein Complex Jigsaw migration

The GitHub repository has already been renamed to:

`https://github.com/219plgit/protein-complex-jigsaw`

This package is an **overlay** for the current repository. It deliberately keeps `src/scijigsaw/` intact for backward compatibility and adds `src/pcjigsaw/` as the preferred public namespace.

## Recommended application

From a local checkout of `protein-complex-jigsaw`, copy the contents of this migration package over the repository root, then review and commit:

```bash
cp -R pcj_migration_v4/* /path/to/protein-complex-jigsaw/
cd /path/to/protein-complex-jigsaw
python -m pip install -e ".[dev]"
pytest -q
```

Then test both namespaces:

```bash
python -c "import pcjigsaw; print(pcjigsaw.__version__)"
python -c "import scijigsaw; print(scijigsaw.__version__)"
pcjigsaw-count vamp2
scijigsaw-count vamp2
```

Commit suggestion:

```bash
git add README.md pyproject.toml CITATION.cff .zenodo.json \
        CHANGELOG_PCJ_v4.0.0.md src/pcjigsaw tests/test_pcjigsaw_namespace.py
git commit -m "Rename public software to Protein Complex Jigsaw and add perturbation analysis"
git push
```

Create a GitHub release/tag **v4.0.0** after CI passes.

## Why v4.0.0 rather than v1.0.0?

The repository already has a v3.0.0 software history. Resetting the version to v1.0.0 after a rename would make provenance and reproducibility harder to follow. A public namespace change plus a substantial new analysis layer is naturally a major release, so **v4.0.0** is the cleanest continuation.

## Manuscript consistency

The manuscript/package prepared earlier used `v1.0.0` as a provisional release label. Before bioRxiv/PLOS submission, change that label to **v4.0.0** wherever the software release is mentioned, unless you intentionally decide to reset the project history.

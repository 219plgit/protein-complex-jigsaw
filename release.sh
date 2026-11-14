#!/usr/bin/env bash
# Protein Complex Jigsaw v4.1.0 — release gate. Run from the repository root.
# Portable across macOS (BSD) and Linux (GNU).
set -euo pipefail

VERSION="4.1.0"
TAG="v${VERSION}"

# 0. sanity: clean tree, tests, version agreement
git diff --quiet || { echo "uncommitted changes — commit first"; exit 1; }
python -m pytest -q
grep -q "version = \"${VERSION}\"" pyproject.toml || { echo "pyproject version mismatch"; exit 1; }
grep -q "version: \"${VERSION}\"" CITATION.cff || { echo "CITATION.cff version mismatch"; exit 1; }
grep -q "__version__ = \"${VERSION}\"" src/pcjigsaw/__init__.py || { echo "package version mismatch"; exit 1; }

# 1. reproduce the manuscript numbers
python verify_paper.py

# 2. the released wheel must regenerate the teaching material on its own:
#    build it, install it in a clean environment and check every answer-key number
rm -rf dist && python -m pip wheel . --no-deps -w dist -q
VENV="$(mktemp -d)/venv"; RUN="$(mktemp -d)"
python -m venv "${VENV}"
# a clean environment: nothing from the checkout may leak onto the path
env -u PYTHONPATH "${VENV}/bin/pip" install -q dist/protein_complex_jigsaw-${VERSION}-py3-none-any.whl
( cd "${RUN}" && unset PYTHONPATH && "${VENV}/bin/pcjigsaw-teach" check \
              && "${VENV}/bin/pcjigsaw-teach" vamp2 vamp2_pack >/dev/null \
              && cmp vamp2_pack/vamp2_answers.json "${OLDPWD}/education/packs/vamp2/vamp2_answers.json" )
echo "wheel reproduces the teaching answer keys"

# 3. set today's date in CITATION.cff
perl -pi -e "s/^date-released: .*/date-released: $(date +%F)/" CITATION.cff
if ! git diff --quiet CITATION.cff; then
  git add CITATION.cff
  git commit -m "v${VERSION}: set release date"
fi

# 4. annotated tag + push, GitHub release with the wheel attached
git tag -a "${TAG}" -m "Protein Complex Jigsaw ${TAG}: PCJ Flow core, Method -> Tool -> Education"
git push origin main --follow-tags
gh release create "${TAG}" dist/protein_complex_jigsaw-${VERSION}-py3-none-any.whl \
  --title "Protein Complex Jigsaw ${TAG}" \
  --notes-file CHANGELOG_PCJ_FLOW_v4.1.0.md

echo "Done. Release: https://github.com/219plgit/protein-complex-jigsaw/releases/tag/${TAG}"
echo "Zenodo archives the GitHub release automatically once the integration is enabled."

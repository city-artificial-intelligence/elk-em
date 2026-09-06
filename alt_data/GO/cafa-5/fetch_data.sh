#!/usr/bin/env bash
# Retrieves the CAFA 5 competition data required to reproduce Section 5.2.
# The data is not redistributed here; it is obtained directly from Kaggle
# under the competition's own terms, which you must accept on the
# competition page before the download will succeed:
#   https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/rules
#
# Requires the Kaggle CLI:  pip install kaggle
# and an API token at ~/.kaggle/kaggle.json
set -euo pipefail
cd "$(dirname "$0")"
kaggle competitions download -c cafa-5-protein-function-prediction -p .
unzip -o cafa-5-protein-function-prediction.zip -d .
rm cafa-5-protein-function-prediction.zip
shasum -a 256 -c CHECKSUMS.sha256 2>/dev/null || sha256sum -c CHECKSUMS.sha256
echo "CAFA 5 data verified against the checksums used in the paper."

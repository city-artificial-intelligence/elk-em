# ELK-Em

Closure-Aware Embeddings for ℰℒ+⊥ Ontologies.

Reproduction code for the paper's §5.1 (ranking entailed atomic subsumptions
on GALEN / GO / Anatomy) and §5.2 (inductive protein-function prediction on
CAFA5) lives under [paper/](paper/). See [paper/README.md](paper/README.md)
for end-to-end reproduction instructions.

Data ships in [data/](data/) (Box2EL benchmark for §5.1) and
[alt_data/GO/](alt_data/GO/) (normalised GO TBox and precomputed ESM-2
embeddings for §5.2). The CAFA 5 competition files are not redistributed here;
run [alt_data/GO/cafa-5/fetch_data.sh](alt_data/GO/cafa-5/fetch_data.sh) to
obtain them from Kaggle under the competition's terms.

Technical report: CR-ELKEm-TR.pdf

## Licensing

Code in this repository is released under the Apache License 2.0 (see LICENSE).
Datasets under data/ and alt_data/ are third-party works redistributed under
their own terms; see the LICENSE.txt or ACKNOWLEDGEMENTS.txt file in each
dataset directory. CAFA 5 competition data is not redistributed — see
alt_data/GO/cafa-5/fetch_data.sh.

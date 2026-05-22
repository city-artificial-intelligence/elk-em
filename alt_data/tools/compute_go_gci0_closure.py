"""
alt_data/tools/compute_go_gci0_closure.py

Computes the full ELK-based GCI0 deductive closure for a GO OWL file and
extracts it to DELEclosure/nf1.npy for use in propagate_go_annotations.

Steps:
  1. Run ELK reasoner on go.owl → materialise all C ⊑ D pairs → gci0_ontology.owl
  2. Parse gci0_ontology.owl + classes.json → DELEclosure/nf1.npy  (N,2) int32

Outputs (written inside --data-dir):
  gci0_ontology.owl
  DELEclosure/nf1.npy
  DELEclosure/skipped/nf1.txt

Usage:
    python alt_data/tools/compute_go_gci0_closure.py --data-dir alt_data/GO
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Step 1 — ELK GCI0 deductive closure
# (requires mowl + JVM; mowl.init_jvm must be called before import)
# ---------------------------------------------------------------------------

def compute_gci0_closure(data_dir: str, ontology_file: str, jvm_heap: str) -> None:
    import mowl
    mowl.init_jvm(jvm_heap)

    from java.util import HashSet
    from mowl.datasets import PathDataset
    from mowl.owlapi import OWLAPIAdapter
    from mowl.owlapi.defaults import BOT, TOP
    from org.semanticweb.elk.owlapi import ElkReasonerFactory
    from org.semanticweb.owlapi.model import IRI
    from uk.ac.manchester.cs.owl.owlapi import OWLSubClassOfAxiomImpl

    ontology_path = os.path.join(data_dir, ontology_file)
    print(f'[gci0] Loading {ontology_path}', flush=True)
    dataset = PathDataset(ontology_path)

    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager

    reasoner_factory = ElkReasonerFactory()
    reasoner = reasoner_factory.createReasoner(dataset.ontology)

    all_classes = list(dataset.ontology.getClassesInSignature())
    print(f'[gci0] Running ELK over {len(all_classes)} classes', flush=True)

    new_axioms = HashSet()
    for i, cl in enumerate(all_classes):
        superclasses = list(reasoner.getSuperClasses(cl, False).getFlattened())
        subclasses   = list(reasoner.getSubClasses(cl, False).getFlattened())

        sub = adapter.create_class(BOT) if 'Nothing' in str(cl) else (
              adapter.create_class(TOP) if 'Thing'   in str(cl) else cl)

        for sup in superclasses:
            sup_ = adapter.create_class(BOT) if 'Nothing' in str(sup) else (
                   adapter.create_class(TOP) if 'Thing'   in str(sup) else sup)
            new_axioms.add(OWLSubClassOfAxiomImpl(sub, sup_, []))

        for child in subclasses:
            child_ = adapter.create_class(BOT) if 'Nothing' in str(child) else (
                     adapter.create_class(TOP) if 'Thing'   in str(child) else child)
            new_axioms.add(OWLSubClassOfAxiomImpl(child_, sub, []))

        if (i + 1) % 1000 == 0 or (i + 1) == len(all_classes):
            print(f'[gci0]   {i+1}/{len(all_classes)}', flush=True)

    out_ont = adapter.create_ontology('http://gci0_ontology')
    manager.addAxioms(out_ont, new_axioms)

    out_path = Path(data_dir) / 'gci0_ontology.owl'
    output_uri = out_path.resolve().as_uri()
    manager.saveOntology(out_ont, IRI.create(output_uri))
    print(f'[gci0] Wrote {out_path}  ({new_axioms.size()} axioms)', flush=True)


# ---------------------------------------------------------------------------
# Step 2 — Parse gci0_ontology.owl → DELEclosure/nf1.npy
# (pure Python + numpy, no JVM)
# ---------------------------------------------------------------------------

_OWL_NS = 'http://www.w3.org/2002/07/owl#'


def _iri_to_key(iri: str) -> str:
    """Map a raw IRI to the key format used in classes.json.

    OBO IRIs (e.g. http://purl.obolibrary.org/obo/GO_0000001) are kept as-is
    since classes.json stores full IRIs. OWL builtins become 'owl:Nothing' /
    'owl:Thing' to match the classes.json convention.
    """
    if iri.startswith(_OWL_NS):
        return 'owl:' + iri[len(_OWL_NS):]
    return iri


def _parse_gci0(owl_path: str):
    """Yield (C_key, D_key) from C ⊑ D axioms in RDF/XML OWL.

    Keys match the full-IRI format used in classes.json.
    """
    about_re = re.compile(r'<owl:Class rdf:about="([^"]+)"')
    subof_re  = re.compile(r'<rdfs:subClassOf rdf:resource="([^"]+)"')
    current = None
    with open(owl_path) as f:
        for line in f:
            m = about_re.search(line)
            if m:
                current = _iri_to_key(m.group(1))
                continue
            if current is None:
                continue
            m = subof_re.search(line)
            if m:
                yield current, _iri_to_key(m.group(1))


def extract_nf1(data_dir: str) -> None:
    import numpy as np

    gci0_owl     = os.path.join(data_dir, 'gci0_ontology.owl')
    classes_file = os.path.join(data_dir, 'bins', 'classes.json')
    out_dir      = os.path.join(data_dir, 'DELEclosure')
    skip_dir     = os.path.join(out_dir, 'skipped')
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(skip_dir, exist_ok=True)

    with open(classes_file) as f:
        classes: dict[str, int] = json.load(f)

    pairs, skipped = [], []
    for c, d in _parse_gci0(gci0_owl):
        if c in classes and d in classes:
            pairs.append((classes[c], classes[d]))
        else:
            skipped.append(f'{c}  {d}')

    arr = (np.array(pairs, dtype=np.int32) if pairs
           else np.empty((0, 2), dtype=np.int32))
    nf1_path = os.path.join(out_dir, 'nf1.npy')
    np.save(nf1_path, arr)
    print(f'[nf1] kept {len(pairs):,}  skipped {len(skipped):,}', flush=True)
    print(f'[nf1] wrote {nf1_path}', flush=True)

    skip_path = os.path.join(skip_dir, 'nf1.txt')
    with open(skip_path, 'w') as f:
        f.write('\n'.join(skipped) + '\n' if skipped else '')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Compute GO GCI0 deductive closure.')
    p.add_argument('--data-dir',      default='alt_data/GO',
                   help='Directory containing go.owl and bins/classes.json')
    p.add_argument('--ontology-file', default='go.owl')
    p.add_argument('--jvm-heap',      default=None,
                   help='JVM heap size (default: SEMELEL_JVM_HEAP env or 150g)')
    p.add_argument('--skip-elk',      action='store_true',
                   help='Skip ELK step (gci0_ontology.owl already exists)')
    return p.parse_args()


def main():
    args = parse_args()

    jvm_heap = (args.jvm_heap
                or os.environ.get('SEMELEL_JVM_HEAP', '150g'))

    if not args.skip_elk:
        print(f'[main] Step 1: ELK closure  (heap={jvm_heap})', flush=True)
        compute_gci0_closure(args.data_dir, args.ontology_file, jvm_heap)
    else:
        print('[main] Skipping ELK step', flush=True)

    print('[main] Step 2: extract nf1.npy', flush=True)
    extract_nf1(args.data_dir)
    print('[main] Done.', flush=True)


if __name__ == '__main__':
    main()

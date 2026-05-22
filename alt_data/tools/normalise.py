#!/usr/bin/env python3
"""Normalize an OWL ontology and save aligned .norm and normalized .owl outputs."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import mowl

NORMAL_FORM_KEYS = ["gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot"]
ELPLUSPLUS_ROLE_PREFIXES = (
    "ReflexiveObjectProperty(",
    "SubObjectPropertyOf(",
    "TransitiveObjectProperty(",
)

OWL_PREFIXES = """Prefix(owl:=<http://www.w3.org/2002/07/owl#>)
Prefix(rdf:=<http://www.w3.org/1999/02/22-rdf-syntax-ns#>)
Prefix(xml:=<http://www.w3.org/XML/1998/namespace>)
Prefix(xsd:=<http://www.w3.org/2001/XMLSchema#>)
Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize an OWL ontology.")
    parser.add_argument(
        "--input",
        default="data/go.owl",
        help="Path to the input OWL ontology.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .norm file path. Defaults to the input path with a .norm suffix.",
    )
    parser.add_argument(
        "--owl-output",
        default=None,
        help="Output normalized OWL path. Defaults to <input>.normalized.owl.",
    )
    parser.add_argument(
        "--ontology-iri",
        default="http://example.org/normalized-ontology",
        help="Ontology IRI to use for the saved normalized OWL file.",
    )
    parser.add_argument(
        "--jvm-memory",
        default="8g",
        help="JVM heap size passed to mOWL, for example 4g or 8g.",
    )
    return parser.parse_args()


def collect_normalized_axioms(normalized: dict) -> tuple[list[str], dict[str, int]]:
    counts: dict[str, int] = {}
    axioms_out: list[str] = []

    for key in NORMAL_FORM_KEYS:
        axioms = normalized.get(key, [])
        counts[key] = len(axioms)
        for axiom in axioms:
            owl_axiom = getattr(axiom, "owl_axiom", axiom)
            axioms_out.append(str(owl_axiom))

    return axioms_out, counts


def canonicalize_role_axiom(axiom_str: str) -> str:
    if axiom_str.startswith("TransitiveObjectProperty("):
        match = re.fullmatch(r"TransitiveObjectProperty\((<[^>]+>)\)", axiom_str)
        if match:
            role = match.group(1)
            return f"SubObjectPropertyOf(ObjectPropertyChain({role} {role}) {role})"
    return axiom_str


def _collect_role_axioms_direct(ontology) -> tuple[list[str], dict[str, int]]:
    counts: Counter[str] = Counter()
    axioms_out: list[str] = []

    for imported_ontology in ontology.getImportsClosure():
        for axiom in imported_ontology.getAxioms():
            axiom_str = str(axiom)
            matched_prefix = next(
                (prefix for prefix in ELPLUSPLUS_ROLE_PREFIXES if axiom_str.startswith(prefix)),
                None,
            )
            if matched_prefix is None:
                continue

            counts[matched_prefix[:-1]] += 1
            axioms_out.append(canonicalize_role_axiom(axiom_str))

    return unique_preserve_order(axioms_out), dict(counts)


def collect_elplusplus_role_axioms(ontology) -> tuple[list[str], dict[str, int]]:
    try:
        from java.util import HashSet
        from de.tudresden.inf.lat.jcel.ontology.normalization import OntologyNormalizer
        from de.tudresden.inf.lat.jcel.ontology.axiom.extension import IntegerOntologyObjectFactoryImpl
        from de.tudresden.inf.lat.jcel.owlapi.translator import ReverseAxiomTranslator, Translator

        counts: Counter[str] = Counter()
        axioms_out: list[str] = []

        translator = Translator(
            ontology.getOWLOntologyManager().getOWLDataFactory(),
            IntegerOntologyObjectFactoryImpl(),
        )

        axioms = HashSet()
        axioms.addAll(ontology.getAxioms())
        translator.getTranslationRepository().addAxiomEntities(ontology)

        for imported_ontology in ontology.getImportsClosure():
            axioms.addAll(imported_ontology.getAxioms())
            translator.getTranslationRepository().addAxiomEntities(imported_ontology)

        int_axioms = translator.translateSA(axioms)
        normalized_ontology = OntologyNormalizer().normalize(
            int_axioms, IntegerOntologyObjectFactoryImpl()
        )
        reverse_translator = ReverseAxiomTranslator(translator, ontology)

        for axiom in normalized_ontology:
            try:
                axiom_str = str(reverse_translator.visit(axiom))
            except Exception:
                continue

            matched_prefix = next(
                (prefix for prefix in ELPLUSPLUS_ROLE_PREFIXES if axiom_str.startswith(prefix)),
                None,
            )
            if matched_prefix is None:
                continue

            counts[matched_prefix[:-1]] += 1
            axioms_out.append(canonicalize_role_axiom(axiom_str))

        return unique_preserve_order(axioms_out), dict(counts)
    except Exception as exc:
        print(f"Warning: falling back to direct role axiom extraction: {exc}")
        return _collect_role_axioms_direct(ontology)


def unique_preserve_order(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_lines: list[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)
    return unique_lines


def build_functional_syntax_document(axioms: list[str], ontology_iri: str) -> str:
    body = "\n".join(axioms)
    return f"{OWL_PREFIXES}Ontology(<{ontology_iri}>\n{body}\n)\n"


def normalize_to_norm_file(ontology_path: str | Path, norm_output_path: str | Path) -> None:
    """Normalise an OWL ontology and write only the .norm file (no .normalized.owl).

    Uses the same pipeline as main(): ELNormalizer output for all GCI normal forms
    plus EL++ role axioms (with transitivity rewritten as role chains).

    :param ontology_path: path to the input OWL ontology
    :param norm_output_path: path to write the .norm file
    """
    from mowl.datasets import PathDataset
    from mowl.ontology.normalize import ELNormalizer

    input_path = Path(ontology_path)
    output_path = Path(norm_output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input ontology not found: {input_path}")

    dataset = PathDataset(str(input_path))
    ontology = dataset.ontology
    normalizer = ELNormalizer()
    normalized = normalizer.normalize(ontology)

    normalized_lines, normalized_counts = collect_normalized_axioms(normalized)
    role_lines, role_counts = collect_elplusplus_role_axioms(ontology)
    all_lines = unique_preserve_order(normalized_lines + role_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for line in all_lines:
            fout.write(f"{line}\n")

    print(f"Input ontology: {input_path}")
    print(f"Output .norm file: {output_path}")
    print("\nNormalized TBox counts:")
    for key in NORMAL_FORM_KEYS:
        print(f"- {key}: {normalized_counts[key]}")
    print(f"\nTotal normalized TBox axioms: {len(normalized_lines)}")
    print("Normalized EL++ role axioms appended:")
    if role_counts:
        for key, value in role_counts.items():
            if key == "TransitiveObjectProperty":
                print(f"- {key} (saved as role chain): {value}")
            else:
                print(f"- {key}: {value}")
    else:
        print("- none")
    print(f"\nFinal unique axioms written: {len(all_lines)}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".norm")
    owl_output_path = (
        Path(args.owl_output)
        if args.owl_output
        else input_path.with_name(f"{input_path.stem}.normalized.owl")
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input ontology not found: {input_path}")

    if not mowl.__dict__.get("init_jvm"):
        raise RuntimeError("mOWL is not available in the current Python environment.")

    try:
        import jpype

        if not jpype.isJVMStarted():
            mowl.init_jvm(args.jvm_memory)
    except ImportError:
        mowl.init_jvm(args.jvm_memory)

    from mowl.datasets import PathDataset
    from mowl.ontology.normalize import ELNormalizer

    dataset = PathDataset(str(input_path))
    ontology = dataset.ontology
    normalizer = ELNormalizer()
    normalized = normalizer.normalize(ontology)

    normalized_lines, normalized_counts = collect_normalized_axioms(normalized)
    role_lines, role_counts = collect_elplusplus_role_axioms(ontology)
    all_lines = unique_preserve_order(normalized_lines + role_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for line in all_lines:
            fout.write(f"{line}\n")

    from org.semanticweb.owlapi.apibinding import OWLManager
    from org.semanticweb.owlapi.io import StringDocumentSource
    from org.semanticweb.owlapi.formats import RDFXMLDocumentFormat
    from org.semanticweb.owlapi.model import IRI

    owl_output_path.parent.mkdir(parents=True, exist_ok=True)
    functional_document = build_functional_syntax_document(all_lines, args.ontology_iri)
    manager = OWLManager.createOWLOntologyManager()
    ontology_from_norm = manager.loadOntologyFromOntologyDocument(
        StringDocumentSource(functional_document)
    )
    manager.saveOntology(
        ontology_from_norm,
        RDFXMLDocumentFormat(),
        IRI.create(owl_output_path.resolve().as_uri()),
    )

    print(f"Input ontology: {input_path}")
    print(f"Output .norm file: {output_path}")
    print(f"Output OWL file: {owl_output_path}")
    print("\nNormalized TBox counts:")
    for key in NORMAL_FORM_KEYS:
        print(f"- {key}: {normalized_counts[key]}")

    print(f"\nTotal normalized TBox axioms: {len(normalized_lines)}")
    print("Normalized EL++ role axioms appended:")
    if role_counts:
        for key, value in role_counts.items():
            if key == "TransitiveObjectProperty":
                print(f"- {key} (saved as role chain): {value}")
            else:
                print(f"- {key}: {value}")
    else:
        print("- none")

    print(f"\nFinal unique axioms written: {len(all_lines)}")


if __name__ == "__main__":
    main()

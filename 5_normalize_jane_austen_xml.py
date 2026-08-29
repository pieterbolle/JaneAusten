#!/usr/bin/env python3
"""Normalize speaker/addressee labels in Jane Austen annotation XML.

Safety principles:
- never modifies the source file in place;
- stores original labels as S_raw/A_raw when a change is made;
- changes exact aliases only; ambiguous or unknown labels remain untouched;
- writes a JSONL audit and a JSON summary;
- preserves all text and nested interruption elements.
"""
import argparse
import copy
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input_xml", help="Annotated XML input file")
    p.add_argument("--dictionary", default="jane_austen_character_normalization_v2.json")
    p.add_argument("--output", help="Output XML path; default adds _normalized")
    p.add_argument("--book", help="Override the <book title> used for dictionary selection")
    p.add_argument("--case-sensitive", action="store_true")
    p.add_argument("--no-raw-attributes", action="store_true")
    return p.parse_args()


def lookup_table(mapping, case_sensitive):
    if case_sensitive:
        return mapping
    return {key.casefold(): value for key, value in mapping.items()}


def normalize_value(value, table, case_sensitive):
    key = value if case_sensitive else value.casefold()
    return table.get(key)


def atomic_write_xml(root, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=output.parent, suffix=".tmp") as f:
        temp = Path(f.name)
    try:
        ET.ElementTree(root).write(temp, encoding="utf-8", xml_declaration=True,
                                   short_empty_elements=False)
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)


def main():
    args = parse_args()
    source = Path(args.input_xml)
    output = Path(args.output) if args.output else source.with_name(source.stem + "_normalized.xml")
    audit_path = output.with_suffix(".normalization_audit.jsonl")
    summary_path = output.with_suffix(".normalization_summary.json")

    dictionary = json.loads(Path(args.dictionary).read_text(encoding="utf-8"))
    root = ET.parse(source).getroot()
    book = args.book or root.get("title")
    if book not in dictionary:
        raise KeyError(f"No dictionary found for book title: {book!r}")
    table = lookup_table(dictionary[book], args.case_sensitive)

    changes = []
    unresolved = Counter()
    changed_by_attribute = Counter()
    turn_count = 0

    for turn in root.iter("turn"):
        turn_count += 1
        turn_number = turn.get("number")
        for attribute in ("S", "A"):
            raw = turn.get(attribute)
            if raw is None:
                continue
            canonical = normalize_value(raw, table, args.case_sensitive)
            if canonical is None:
                unresolved[(attribute, raw)] += 1
                continue
            if canonical == raw:
                continue
            if not args.no_raw_attributes:
                raw_name = attribute + "_raw"
                if raw_name not in turn.attrib:
                    turn.set(raw_name, raw)
            turn.set(attribute, canonical)
            changed_by_attribute[attribute] += 1
            changes.append({
                "book": book,
                "turn_number": turn_number,
                "attribute": attribute,
                "raw": raw,
                "canonical": canonical,
            })

    root.set("character_normalization", "exact_alias_dictionary_v2")
    root.set("character_normalization_dictionary", Path(args.dictionary).name)
    atomic_write_xml(root, output)

    with audit_path.open("w", encoding="utf-8") as f:
        for change in changes:
            f.write(json.dumps(change, ensure_ascii=False) + "\n")

    summary = {
        "input": str(source),
        "output": str(output),
        "book": book,
        "turns": turn_count,
        "changed_labels": len(changes),
        "changed_by_attribute": dict(changed_by_attribute),
        "unresolved_labels": [
            {"attribute": attr, "value": value, "count": count}
            for (attr, value), count in unresolved.most_common()
        ],
        "audit": str(audit_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

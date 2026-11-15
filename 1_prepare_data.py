# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Pieter Bolle
import json
import os
import re
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

TURN_NUMBER_PATTERN = re.compile(
    r"<turn\b[^>]*\s+n\s*=\s*[\"'][^\"']*[\"']",
    flags=re.IGNORECASE,
)

def clean_xml_entities(text_str):
    """
    Vertaal HTML/XML entities terug naar pure leestekens, zodat 
    Llama 3 leert werken met echte aanhalingstekens en apostrofs.
    """
    if not text_str: return ""
    text_str = text_str.replace("&quot;", '"')
    text_str = text_str.replace("&#x27;", "'")
    text_str = text_str.replace("&amp;", "&")
    return text_str

def remove_turn_numbers(paragraph):
    """Remove the n attribute from every turn element in a paragraph."""
    removed_count = 0

    for turn in paragraph.iter("turn"):
        if "n" in turn.attrib:
            del turn.attrib["n"]
            removed_count += 1

    return removed_count

def parse_xml_to_llama3_dataset(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    dataset = []
    
    # We slaan de ruwe tekst van de voorgaande paragrafen op als context
    chronological_history = []
    removed_turn_numbers = 0
    paragraphs_with_turns = 0
    
    system_prompt = (
        "You are an expert NLP system specialized in 19th-century English literature. "
        "Your task is to annotate a given raw paragraph with structured XML dialogue tags. "
        "Precisely identify the Speaker (S), Addressee (A), Bystanders (yes/no), and Indirect Speech (yes). "
        "Wrap any dialogue in <turn> tags, and wrap narrative interruptions inside the dialogue in <interruption> tags. "
        "If the paragraph contains no dialogue, output it exactly as is. "
        "Rely strictly on the textual structure and punctuation provided. Do not use prior knowledge of the novel."
    )

    for p in root.findall("p"):
        # 1. Bepaal de 'Target XML' (de output die de assistent moet genereren)
        # We pakken de interne structuur van de <p> tag.
        paragraph_turns = list(p.iter("turn"))
        if paragraph_turns:
            paragraphs_with_turns += 1
        removed_turn_numbers += remove_turn_numbers(p)
        inner_xml_parts = [p.text or ""]
        for child in p:
            inner_xml_parts.append(ET.tostring(child, encoding="unicode", method="xml"))
        
        target_xml = "".join(inner_xml_parts).strip()
        target_xml = clean_xml_entities(target_xml)
        
        # 2. Bepaal de 'Raw Text' (de input die de gebruiker geeft)
        # Dit is puur de tekst, waarbij alle XML-tags (zoals <turn> en <interruption>) gestript zijn.
        raw_text = "".join(p.itertext()).strip()
        raw_text = clean_xml_entities(raw_text)
        
        if not raw_text:
            continue
            
        # Het cijfer hieronder duidt het aantal paragrafen aan.
        context_window = chronological_history[-3:]
        if context_window:
            context_str = "\n\n".join(context_window)
        else:
            context_str = "No preceding context (Start of book)."
            
        user_input = f"Context history:\n{context_str}\n\nTarget paragraph to parse:\n{raw_text}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": target_xml}
        ]
        dataset.append({"messages": messages})
        
        # 4. Voeg deze ruwe paragraaf toe aan de geschiedenis voor de volgende beurt
        chronological_history.append(raw_text)

    print(
        f"Removed {removed_turn_numbers} turn-number attributes "
        f"from {paragraphs_with_turns} paragraphs containing turns."
    )
    return dataset

def validate_dataset_has_no_turn_numbers(dataset, dataset_name):
    """Validate that no assistant target contains a numbered turn element."""
    offending_examples = []

    for example_index, example in enumerate(dataset):
        assistant_target = example["messages"][-1]["content"]
        if TURN_NUMBER_PATTERN.search(assistant_target):
            offending_examples.append(example_index)

    if offending_examples:
        raise ValueError(
            f"{dataset_name} contains turn numbers in examples: "
            f"{offending_examples[:20]}"
        )

    print(
        f"Validated {len(dataset)} {dataset_name} examples: "
        "no turn-number attributes remain."
    )

if __name__ == "__main__":
    print("Inladen en parsen van de schone XML naar paragraaf-niveau...")
    dataset = parse_xml_to_llama3_dataset("output/pride_and_prejudice_clean.xml")
    
    print(f"Splitsen van {len(dataset)} paragrafen in training en validatie...")
    # random_state zorgt voor een vaste split, shuffle is belangrijk zodat het model 
    # alle fases van het boek ziet tijdens de training
    train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)
    
    validate_dataset_has_no_turn_numbers(train_data, "training")
    validate_dataset_has_no_turn_numbers(val_data, "validation")
    os.makedirs("intermediate_output", exist_ok=True)
    
    with open("intermediate_output/train.jsonl", "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")
            
    with open("intermediate_output/val.jsonl", "w", encoding="utf-8") as f:
        for entry in val_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Data succesvol klaargezet! {len(train_data)} training en {len(val_data)} validatie samples klaar voor Llama 3.")

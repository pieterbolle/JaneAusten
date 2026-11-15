# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Pieter Bolle
import os
import re
import pandas as pd
import html

def safe_xml_text(text):
    """Escaped alleen de karakters die de XML breken, behoudt aanhalingstekens voor de tokenizer."""
    if not text: return ""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def standardize_names(name_string):
    """Schoont namen op en sorteert ze voor consistente evaluatie."""
    if pd.isnull(name_string):
        return "Unknown"
    name_str = str(name_string).strip()
    if name_str in ["", "?", "nan", "Unknown"]:
        return "Unknown"
    names = [n.strip() for n in name_str.split('+') if n.strip()]
    if not names:
        return "Unknown"
    return " + ".join(sorted(names))

def reconstruct_turn_text(text_str, is_indirect):
    """Vormt de ruwe spraaktekst om naar zuivere XML interrupties."""
    text_str = re.sub(r'\s+', ' ', text_str).strip()
    if not text_str:
        return ""
        
    xml_parts = []
    
    if is_indirect:
        # Bij indirecte rede (geen aanhalingstekens) is alles pure gesproken (vertelde) tekst
        xml_parts.append(safe_xml_text(text_str))
    else:
        # Bij directe rede maken we gebruik van de aanhalingstekens om interrupties te taggen
        quote_parts = text_str.split('"')
        for q_idx, q_part in enumerate(quote_parts):
            q_part_stripped = q_part.strip()
            if not q_part_stripped:
                continue
            if q_idx % 2 == 0:  # Buiten aanhalingstekens -> Interruptie/Verteller
                if any(c.isalpha() for c in q_part_stripped):
                    xml_parts.append(f"<interruption>{safe_xml_text(q_part_stripped)}</interruption>")
                else:
                    # Pure leestekens (zoals een komma na een quote) blijven plat
                    xml_parts.append(safe_xml_text(q_part_stripped))
            else:  # Binnen aanhalingstekens -> Directe spraak
                xml_parts.append(f'"{safe_xml_text(q_part_stripped)}"')
                
    return " ".join(xml_parts)

def parse_to_paragraphs(df):
    """
    Slimme paragraaf-parser: groepeert tekst en beurten correct BINNEN 
    dezelfde paragraaf-container totdat er een lege regel verschijnt.
    """
    paragraphs = []
    current_paragraph = []
    
    in_turn = False
    current_turn = None
    
    def flush_paragraph():
        # Sluit de huidige paragraaf af en start een nieuwe
        if current_paragraph:
            paragraphs.append(current_paragraph.copy())
            current_paragraph.clear()

    for idx, row in df.iterrows():
        text_val = str(row['text']).strip() if pd.notnull(row['text']) else ""
        
        has_meta = pd.notnull(row['speaker']) or pd.notnull(row['addressee']) or pd.notnull(row['indirect speech'])
        row_meta = {
            'speaker': row['speaker'],
            'addressee': row['addressee'],
            'by-stander': row['by-stander'],
            'indirect speech': row['indirect speech']
        }
        
        # Een lege regel in Excel sluit een doorlopende paragraaf af (tenzij we middenin een turn zitten)
        if not text_val and not has_meta:
            if not in_turn:
                flush_paragraph()
            continue

        # Splits op $ en $$ maar BEHOUD de leestekens
        parts = re.split(r'(\$\$|\$)', text_val)
        
        meta_applied = False
        if has_meta and in_turn and current_turn is not None:
            for k, v in row_meta.items():
                if pd.notnull(v): current_turn[k] = v
            meta_applied = True

        for part in parts:
            if not part: continue
            
            if part == '$':
                in_turn = True
                current_turn = {
                    "type": "turn",
                    "text_parts": [],
                    "speaker": None,
                    "addressee": None,
                    "by-stander": None,
                    "indirect speech": None
                }
                if has_meta and not meta_applied:
                    for k, v in row_meta.items():
                        if pd.notnull(v): current_turn[k] = v
                    meta_applied = True
                    
            elif part == '$$':
                if in_turn and current_turn:
                    # Voeg de afgeronde turn toe AAN DE PARAGRAAF
                    current_paragraph.append(current_turn)
                in_turn = False
                current_turn = None
                
            else:
                if in_turn and current_turn is not None:
                    current_turn["text_parts"].append(part.strip())
                else:
                    if part.strip():
                        # Losse narratieve tekst behoort direct tot de paragraaf
                        current_paragraph.append({"type": "narrative", "text": part.strip()})
                        
    # Einde bestand opruimen
    flush_paragraph()
    if in_turn and current_turn:
        current_paragraph.append(current_turn)
        paragraphs.append(current_paragraph)
        
    return paragraphs

def generate_clean_xml(csv_path, output_xml_path):
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    paragraphs = parse_to_paragraphs(df)
    
    xml_lines = ['<?xml version="1.0" encoding="utf-8"?>']
    xml_lines.append('<book title="Pride and Prejudice" author="Jane Austen">')
    turn_counter = 1
    
    for p in paragraphs:
        p_xml_parts = []
        for item in p:
            if item["type"] == "narrative":
                # Narratieve tekst netjes escapen
                clean_text = safe_xml_text(item["text"])
                p_xml_parts.append(clean_text)
            else:
                # Turn metadata inladen
                speaker = standardize_names(item.get('speaker'))
                addressee = standardize_names(item.get('addressee'))
                
                raw_bystander = str(item.get('by-stander')).strip() if pd.notnull(item.get('by-stander')) else ""
                has_bystanders = "yes" if raw_bystander not in ["", "?", "nan", "None", "no"] else "no"
                
                raw_indirect = str(item.get('indirect speech')).strip().lower() if pd.notnull(item.get('indirect speech')) else ""
                is_indirect = True if raw_indirect == "yes" else False
                
                # Voeg alle zinsdelen van de turn samen
                combined_text = " ".join(item["text_parts"])
                inner_xml = reconstruct_turn_text(combined_text, is_indirect)
                
                # Bouw attributen en de turn tag
                attr_str = f'n="{turn_counter}" S="{speaker}" A="{addressee}" bystanders="{has_bystanders}"'
                if is_indirect:
                    attr_str += ' indirectspeech="yes"'
                    
                turn_xml = f'<turn {attr_str}>{inner_xml}</turn>'
                p_xml_parts.append(turn_xml)
                turn_counter += 1
                
        # Plak alle onderdelen van deze paragraaf aan elkaar
        combined_p = " ".join(p_xml_parts).strip()
        if combined_p:
            xml_lines.append(f'  <p>{combined_p}</p>')
            
    xml_lines.append('</book>')
    
    flat_xml = "\n".join(xml_lines)
    
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(flat_xml)
        
    print(f"✔ XML succesvol gegenereerd! Totaal aantal beurten (turns): {turn_counter - 1}")

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    # Zorg dat paden kloppen met je eigen mapstructuur
    generate_clean_xml("input/pride_and_prejudice_JK.csv", "output/pride_and_prejudice_clean.xml")

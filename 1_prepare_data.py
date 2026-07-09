import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split

def standardize_names(name_string):
    """
    Trimt ongewenste spaties en sorteert meervoudige namen alfabetisch
    om combinatorische explosie in de evaluatie te voorkomen.
    """
    if pd.isnull(name_string):
        return "Unknown"

    name_str = str(name_string).strip()
    if name_str in ["", "?", "nan", "Unknown"]:
        return "Unknown"
    
    # Splits op de '+', trim de spaties per individu, filter lege strings eruit
    names = [n.strip() for n in name_str.split('+') if n.strip()]
    
    if not names:
        return "Unknown"
        
    # Sorteer alfabetisch en voeg weer samen met een strakke ' + '
    return " + ".join(sorted(names))

def load_and_parse_data(csv_path):
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    turns = []
    
    current_text = []
    current_speaker = None
    current_addressee = None
    current_bystander = "no"
    
    for idx, row in df.iterrows():
        text_val = str(row['text']) if pd.notnull(row['text']) else ""
        
        # Als er een nieuwe spreker gedefinieerd is, start een nieuwe turn
        if pd.notnull(row['speaker']):
            if current_text:
                turns.append({
                    "text": " ".join(current_text),
                    "speaker": current_speaker,
                    "addressee": current_addressee,
                    "bystanders": current_bystander
                })
            current_text = [text_val]

            current_speaker = standardize_names(row['speaker'])
            current_addressee = standardize_names(row['addressee'])
            
            # Bystanders is binair (yes/no), dus die hoeft niet gesorteerd te worden
            current_bystander = "yes" if pd.notnull(row['by-stander']) and row['by-stander'] != "?" else "no"
        else:
            if text_val:
                current_text.append(text_val)
                
        # Check of de turn hier eindigt op basis van de $$ marker
        if "$$" in text_val and current_text:
            turns.append({
                "text": " ".join(current_text),
                "speaker": current_speaker,
                "addressee": current_addressee,
                "bystanders": current_bystander
            })
            current_text = []
    return turns

def build_xml_target(turn):
    cleaned = turn["text"].replace('$$', '').replace('$', '')
    parts = cleaned.split('"')
    reconstructed = ""
    for i, part in enumerate(parts):
        if i % 2 == 0:  # Buiten aanhalingstekens (interruptie door verteller)
            if part.strip() and any(c.isalpha() for c in part):
                reconstructed += f"<interruption>{part}</interruption>"
            else:
                reconstructed += part
        else:  # Binnen aanhalingstekens (directe rede)
            reconstructed += f'"{part}"'
            
    return f'<turn S="{turn["speaker"]}" A="{turn["addressee"]}" bystanders="{turn["bystanders"]}">{reconstructed}</turn>'

def convert_to_llama3_format(turns):
    dataset = []
    system_prompt = (
        "You are an expert NLP system. Convert the raw narrative literary text into a structured XML turn format. "
        "Identify the Speaker (S), Addressee (A), Bystanders (yes/no), and wrap narrative interruptions within the turn in <interruption> tags."
    )
    
    # We geven het model steeds een klein venster van de context mee (bijv. 20 voorgaande turns)
    for i in range(len(turns)):
        context_str = "\n".join([t["text"] for t in turns[max(0, i-20):i]])
        user_input = f"Context history:\n{context_str}\n\nTarget text to parse:\n{turns[i]['text']}"
        target_xml = build_xml_target(turns[i])
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": target_xml}
        ]
        dataset.append({"messages": messages})
    return dataset

if __name__ == "__main__":
    raw_turns = load_and_parse_data("input/pride_and_prejudice_JK.csv")
    formatted_data = convert_to_llama3_format(raw_turns)
    
    train_data, val_data = train_test_split(formatted_data, test_size=0.1, random_state=42)
    
    with open("intermediate_output/train.jsonl", "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")
            
    with open("intermediate_output/val.jsonl", "w") as f:
        for entry in val_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Data succesvol klaargezet! {len(train_data)} training en {len(val_data)} validatie samples.")

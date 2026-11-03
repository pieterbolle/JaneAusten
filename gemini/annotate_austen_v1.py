#!/usr/bin/env python3
"""
annotate_austen.py
------------------
Annoteert een roman van Jane Austen (.txt) naar het XML-formaat van
pride_and_prejudice_clean.xml, inclusief:
  - Detectie van directe rede, indirecte rede (indirectspeech="yes") en vrije indirecte rede.
  - Aanduiding van <interruption> binnen een <turn>.
  - Expliciete lijst van aanwezige bystanders (gescheiden door " + ", of "no" indien niemand).
  - Automatische normalisatie van alle personagenamen (S, A en bystanders) naar één
    canonieke naam per personage (bv. Lizzy / Eliza / Elizabeth -> Mrs. Elizabeth Bennet).

Gebruik (1 verplichte parameter: het .txt bestand van het boek):
    python annotate_austen.py pride_and_prejudice.txt
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Optionele import van de officiële google-genai SDK (valt automatisch terug op
# standaardbibliotheek urllib.request indien google-genai niet geïnstalleerd is).
try:
    from google import genai
    from google.genai import types
    HAS_GENAI_SDK = True
except ImportError:
    HAS_GENAI_SDK = False


# =====================================================================
# 1. JSON Schema's voor Gemini Structured Outputs
# =====================================================================

CHARACTER_REGISTRY_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "book_title": {"type": "STRING"},
        "author": {"type": "STRING"},
        "characters": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "canonical_name": {
                        "type": "STRING",
                        "description": (
                            "De genormaliseerde, canonieke naam van het personage doorheen het hele boek, "
                            "bv. 'Mrs. Elizabeth Bennet', 'Mr. Fitzwilliam Darcy', 'Miss Jane Bennet', "
                            "'Mr. Bennet', 'Mrs. Bennet', 'Miss Elinor Dashwood', 'Miss Marianne Dashwood'."
                        ),
                    },
                    "aliases": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": (
                            "Alle mogelijke roepnamen, bijnamen, titels, meisjesnamen, getrouwde namen en "
                            "beschrijvende referenties die in de tekst voorkomen (bv. 'Lizzy', 'Eliza', "
                            "'Miss Elizabeth Bennet', 'Elizabeth', 'Miss Eliza Bennet')."
                        ),
                    },
                },
                "required": ["canonical_name", "aliases"],
            },
        },
    },
    "required": ["book_title", "author", "characters"],
}

BATCH_ANNOTATION_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "paragraphs": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "paragraph_index": {
                        "type": "INTEGER",
                        "description": "De exacte index (p=...) van de paragraaf uit de input-batch.",
                    },
                    "scene_characters_present": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": (
                            "Lijst van alle personages (in hun canonieke naam) die aan het einde van "
                            "deze paragraaf fysiek aanwezig zijn in de huidige scène/ruimte."
                        ),
                    },
                    "has_speech": {
                        "type": "BOOLEAN",
                        "description": (
                            "True als deze paragraaf directe rede, indirecte rede of vrije indirecte rede "
                            "bevat die met <turn> geannoteerd moet worden."
                        ),
                    },
                    "annotated_xml": {
                        "type": "STRING",
                        "description": (
                            "Indien has_speech=True: de EXACTE tekst van de paragraaf (zonder één enkel woord "
                            "of leesteken te wijzigen) met daarin <turn> en eventueel <interruption> tags. "
                            "Indien has_speech=False: laat leeg ('')."
                        ),
                    },
                },
                "required": [
                    "paragraph_index",
                    "scene_characters_present",
                    "has_speech",
                    "annotated_xml",
                ],
            },
        }
    },
    "required": ["paragraphs"],
}

ALIAS_RESOLUTION_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "resolutions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "raw_name": {"type": "STRING"},
                    "canonical_name": {"type": "STRING"},
                },
                "required": ["raw_name", "canonical_name"],
            },
        }
    },
    "required": ["resolutions"],
}


# =====================================================================
# 2. Basis-woordenboek voor Jane Austen (dynamisch uitgebreid door Gemini)
# =====================================================================

DEFAULT_AUSTEN_NORMALIZATION: Dict[str, Dict[str, str]] = {
    "pride and prejudice": {
        # Elizabeth
        "elizabeth": "Mrs. Elizabeth Bennet",
        "lizzy": "Mrs. Elizabeth Bennet",
        "eliza": "Mrs. Elizabeth Bennet",
        "miss eliza": "Mrs. Elizabeth Bennet",
        "miss eliza bennet": "Mrs. Elizabeth Bennet",
        "miss elizabeth": "Mrs. Elizabeth Bennet",
        "miss elizabeth bennet": "Mrs. Elizabeth Bennet",
        "elizabeth bennet": "Mrs. Elizabeth Bennet",
        "mrs. darcy": "Mrs. Elizabeth Bennet",
        "mrs. elizabeth bennet": "Mrs. Elizabeth Bennet",
        # Jane
        "jane": "Miss Jane Bennet",
        "miss bennet": "Miss Jane Bennet",
        "jane bennet": "Miss Jane Bennet",
        "miss jane bennet": "Miss Jane Bennet",
        "mrs. bingley": "Miss Jane Bennet",
        "the eldest miss bennet": "Miss Jane Bennet",
        # Kitty / Catherine
        "kitty": "Miss Catherine Bennet",
        "catherine": "Miss Catherine Bennet",
        "miss catherine bennet": "Miss Catherine Bennet",
        "kitty bennet": "Miss Catherine Bennet",
        "catherine bennet": "Miss Catherine Bennet",
        # Lydia
        "lydia": "Miss Lydia Bennet",
        "miss lydia": "Miss Lydia Bennet",
        "miss lydia bennet": "Miss Lydia Bennet",
        "lydia bennet": "Miss Lydia Bennet",
        "mrs. wickham": "Miss Lydia Bennet",
        # Mary
        "mary": "Miss Mary Bennet",
        "miss mary bennet": "Miss Mary Bennet",
        "mary bennet": "Miss Mary Bennet",
        # Parents
        "mr. bennet": "Mr. Bennet",
        "her father": "Mr. Bennet",
        "his father": "Mr. Bennet",
        "mrs. bennet": "Mrs. Bennet",
        "her mother": "Mrs. Bennet",
        "mama": "Mrs. Bennet",
        "his wife": "Mrs. Bennet",
        "his lady": "Mrs. Bennet",
        # Bingleys & Hursts
        "mr. bingley": "Mr. Charles Bingley",
        "bingley": "Mr. Charles Bingley",
        "charles": "Mr. Charles Bingley",
        "charles bingley": "Mr. Charles Bingley",
        "mr. charles bingley": "Mr. Charles Bingley",
        "miss bingley": "Miss Caroline Bingley",
        "caroline": "Miss Caroline Bingley",
        "caroline bingley": "Miss Caroline Bingley",
        "miss caroline bingley": "Miss Caroline Bingley",
        "mrs. hurst": "Mrs. Louisa Hurst",
        "louisa": "Mrs. Louisa Hurst",
        "louisa hurst": "Mrs. Louisa Hurst",
        "mr. hurst": "Mr. Hurst",
        "mr.hurst": "Mr. Hurst",
        # Darcys & De Bourghs
        "mr. darcy": "Mr. Fitzwilliam Darcy",
        "darcy": "Mr. Fitzwilliam Darcy",
        "fitzwilliam darcy": "Mr. Fitzwilliam Darcy",
        "mr. fitzwilliam darcy": "Mr. Fitzwilliam Darcy",
        "miss darcy": "Miss Georgiana Darcy",
        "georgiana": "Miss Georgiana Darcy",
        "georgiana darcy": "Miss Georgiana Darcy",
        "miss georgiana darcy": "Miss Georgiana Darcy",
        "lady catherine": "Lady Catherine de Bourgh",
        "lady catherine de bourgh": "Lady Catherine de Bourgh",
        "her ladyship": "Lady Catherine de Bourgh",
        "miss de bourgh": "Miss Anne de Bourgh",
        "anne": "Miss Anne de Bourgh",
        "anne de bourgh": "Miss Anne de Bourgh",
        "colonel fitzwilliam": "Colonel Fitzwilliam",
        "fitzwilliam": "Colonel Fitzwilliam",
        # Lucases & Collins
        "charlotte": "Mrs. Charlotte Collins",
        "charlotte lucas": "Mrs. Charlotte Collins",
        "miss lucas": "Mrs. Charlotte Collins",
        "mrs. collins": "Mrs. Charlotte Collins",
        "mrs. charlotte collins": "Mrs. Charlotte Collins",
        "mr. collins": "Mr. William Collins",
        "william collins": "Mr. William Collins",
        "mr. william collins": "Mr. William Collins",
        "sir william": "Sir William Lucas",
        "sir william lucas": "Sir William Lucas",
        "lady lucas": "Lady Lucas",
        "mrs. lucas": "Lady Lucas",
        "maria": "Miss Maria Lucas",
        "maria lucas": "Miss Maria Lucas",
        "miss maria lucas": "Miss Maria Lucas",
        "a young lucas": "Master Lucas",
        # Others
        "mr. wickham": "Mr. George Wickham",
        "wickham": "Mr. George Wickham",
        "george wickham": "Mr. George Wickham",
        "mr. george wickham": "Mr. George Wickham",
        "mr. denny": "Mr. Denny",
        "denny": "Mr. Denny",
        "colonel forster": "Colonel Forster",
        "mrs. forster": "Mrs. Forster",
        "mr. gardiner": "Mr. Edward Gardiner",
        "mr. edward gardiner": "Mr. Edward Gardiner",
        "mrs. gardiner": "Mrs. Gardiner",
        "mrs. philips": "Mrs. Philips",
        "mr. philips": "Mr. Philips",
        "mrs. reynolds": "Mrs. Reynolds",
        "mrs. hill": "Mrs. Hill",
        "mrs. hills": "Mrs. Hill",
        "hill": "Mrs. Hill",
        "mrs. jenkinson": "Mrs. Jenkinson",
        "mrs. annesley": "Mrs. Annesley",
        "mrs. younge": "Mrs. Younge",
        "miss king": "Miss Mary King",
    },
    "sense and sensibility": {
        "elinor": "Miss Elinor Dashwood",
        "elinor dashwood": "Miss Elinor Dashwood",
        "miss dashwood": "Miss Elinor Dashwood",
        "miss elinor dashwood": "Miss Elinor Dashwood",
        "marianne": "Miss Marianne Dashwood",
        "marianne dashwood": "Miss Marianne Dashwood",
        "marrianne dashwood": "Miss Marianne Dashwood",
        "miss marianne": "Miss Marianne Dashwood",
        "miss marianne dashwood": "Miss Marianne Dashwood",
        "margaret": "Miss Margaret Dashwood",
        "miss margaret": "Miss Margaret Dashwood",
        "margaret dashwood": "Miss Margaret Dashwood",
        "mrs. dashwood": "Mrs. Henry Dashwood",
        "mrs. henry dashwood": "Mrs. Henry Dashwood",
        "mama": "Mrs. Henry Dashwood",
        "mr. john dashwood": "Mr. John Dashwood",
        "john dashwood": "Mr. John Dashwood",
        "john": "Mr. John Dashwood",
        "mr. dashwood": "Mr. John Dashwood",
        "mrs. john dashwood": "Mrs. Fanny Dashwood",
        "fanny": "Mrs. Fanny Dashwood",
        "fanny dashwood": "Mrs. Fanny Dashwood",
        "mrs. dashwood's daughter's wife": "Mrs. Fanny Dashwood",
        "edward": "Mr. Edward Ferrars",
        "edward ferrars": "Mr. Edward Ferrars",
        "mr. edward ferrars": "Mr. Edward Ferrars",
        "mr. ferrars": "Mr. Edward Ferrars",
        "robert": "Mr. Robert Ferrars",
        "robert ferrars": "Mr. Robert Ferrars",
        "mr. robert ferrars": "Mr. Robert Ferrars",
        "mrs. ferrars": "Mrs. Ferrars",
        "willoughby": "Mr. John Willoughby",
        "mr. willoughby": "Mr. John Willoughby",
        "john willoughby": "Mr. John Willoughby",
        "mr. john willoughby": "Mr. John Willoughby",
        "colonel brandon": "Colonel Brandon",
        "brandon": "Colonel Brandon",
        "colonel": "Colonel Brandon",
        "mr. brandon": "Colonel Brandon",
        "sir john": "Sir John Middleton",
        "sir john middleton": "Sir John Middleton",
        "lady middleton": "Lady Middleton",
        "mrs. jennings": "Mrs. Jennings",
        "the good-natured old lady": "Mrs. Jennings",
        "mr. palmer": "Mr. Thomas Palmer",
        "palmer": "Mr. Thomas Palmer",
        "mrs. palmer": "Mrs. Charlotte Palmer",
        "charlotte": "Mrs. Charlotte Palmer",
        "charlotte palmer": "Mrs. Charlotte Palmer",
        "lucy": "Miss Lucy Steele",
        "lucy steele": "Miss Lucy Steele",
        "miss lucy": "Miss Lucy Steele",
        "miss lucy steele": "Miss Lucy Steele",
        "lucy ferrars": "Miss Lucy Steele",
        "miss steele": "Miss Anne Steele",
        "anne steele": "Miss Anne Steele",
        "anne": "Miss Anne Steele",
        "nancy": "Miss Anne Steele",
    },
}


# =====================================================================
# 3. Gemini API Wrapper (ondersteunt zowel google-genai SDK als REST)
# =====================================================================

class GeminiAnnotatorClient:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro") -> None:
        self.api_key = api_key
        self.model_name = model_name
        self._sdk_client = genai.Client(api_key=api_key) if HAS_GENAI_SDK else None

    def generate_structured(
        self,
        prompt: str,
        response_schema: Dict[str, Any],
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Roept het Gemini frontier model aan en geeft het geparsete JSON-object terug."""
        if self._sdk_client is not None:
            cfg = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=temperature,
                system_instruction=system_instruction,
            )
            resp = self._sdk_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=cfg,
            )
            return json.loads(resp.text)

        # Fallback via standaardbibliotheek urllib.request (geen externe packages nodig)
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
        text_out = body["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text_out)


# =====================================================================
# 4. Hulpfuncties voor Tekst- en Gutenberg-verwerking
# =====================================================================

def parse_gutenberg_text(filepath: Path) -> Tuple[str, str, List[str], int, int]:
    """
    Leest een Project Gutenberg .txt bestand, extraheert titel en auteur,
    splitst in paragrafen en bepaalt de indexen van de literaire hoofdtekst.
    """
    raw_text = filepath.read_text(encoding="utf-8-sig", errors="replace")
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    title_match = re.search(r"^Title:\s*(.+)$", raw_text, flags=re.MULTILINE)
    author_match = re.search(r"^Author:\s*(.+)$", raw_text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else filepath.stem.replace("_", " ").title()
    author = author_match.group(1).strip() if author_match else "Jane Austen"

    blocks = re.split(r"\n\s*\n+", raw_text)
    paragraphs: List[str] = []
    for block in blocks:
        cleaned = re.sub(r"\s+", " ", block.strip())
        if cleaned:
            paragraphs.append(cleaned)

    start_idx = 0
    end_idx = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK", p, re.IGNORECASE):
            start_idx = i + 1
        elif re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK", p, re.IGNORECASE):
            end_idx = i
            break

    for i in range(start_idx, min(start_idx + 40, end_idx)):
        if re.match(r"^CHAPTER\s+(I|1)\b", paragraphs[i], re.IGNORECASE) and len(paragraphs[i]) < 30:
            start_idx = i + 1
            break

    return title, author, paragraphs, start_idx, end_idx


# =====================================================================
# 5. Opbouw van het Canonieke Personage-register & Naamnormalisatie
# =====================================================================

def build_character_registry(
    client: GeminiAnnotatorClient,
    title: str,
    author: str,
    sample_paragraphs: List[str],
) -> Dict[str, str]:
    """
    Bouwt een alias -> canonical_name mapping voor het volledige boek.
    Combineert de ingebouwde Austen-defaults met een door Gemini gegenereerd
    personage-register op basis van het boek.
    """
    alias_map: Dict[str, str] = {}

    title_key = title.lower().strip()
    for known_title, mapping in DEFAULT_AUSTEN_NORMALIZATION.items():
        if known_title in title_key:
            for k, v in mapping.items():
                alias_map[k.lower().strip()] = v
                alias_map[v.lower().strip()] = v

    sample_text = "\n".join(sample_paragraphs[:150])[:25000]
    prompt = f"""Je bent een literair expert in het oeuvre van {author}.
We gaan het boek "{title}" van {author} annoteren.
Maak een uitputtende lijst van ALLE personages in "{title}" (zowel hoofd- als bijpersonages).

Geef voor elk personage:
1. `canonical_name`: Eén unieke, formele genormaliseerde naam (bv. voor Pride and Prejudice:
   'Mrs. Elizabeth Bennet' voor Elizabeth/Lizzy/Eliza, 'Miss Jane Bennet', 'Miss Catherine Bennet',
   'Miss Lydia Bennet', 'Miss Mary Bennet', 'Mr. Bennet', 'Mrs. Bennet', 'Mr. Fitzwilliam Darcy',
   'Mr. Charles Bingley', 'Miss Caroline Bingley', 'Mrs. Louisa Hurst', 'Mr. Hurst',
   'Mr. William Collins', 'Mrs. Charlotte Collins', 'Mr. George Wickham', 'Lady Catherine de Bourgh',
   'Miss Anne de Bourgh', 'Colonel Fitzwilliam', 'Sir William Lucas', 'Lady Lucas', 'Miss Maria Lucas',
   'Mr. Edward Gardiner', 'Mrs. Gardiner', 'Mrs. Philips', 'Mr. Philips', enz.).
2. `aliases`: Alle mogelijke varianten, voornamen, bijnamen, familienamen, titels voor en na een
   huwelijk (bv. 'Miss Lucas' en 'Mrs. Collins'), en relationele aanduidingen.

Begin van de tekst ter referentie:
{sample_text}
"""
    try:
        data = client.generate_structured(
            prompt=prompt,
            response_schema=CHARACTER_REGISTRY_SCHEMA,
            temperature=0.1,
        )
        for char in data.get("characters", []):
            canon = char.get("canonical_name", "").strip()
            aliases = char.get("aliases", [])
            if not canon:
                continue
            if canon.lower() in alias_map:
                canon = alias_map[canon.lower()]
            else:
                for al in aliases:
                    if al.lower().strip() in alias_map:
                        canon = alias_map[al.lower().strip()]
                        break
            alias_map[canon.lower()] = canon
            for al in aliases:
                al_clean = al.lower().strip()
                if al_clean and al_clean not in alias_map:
                    alias_map[al_clean] = canon
    except Exception as e:
        print(f"[Waarschuwing] Kon dynamisch personageregister niet ophalen ({e}); gebruik defaults.", file=sys.stderr)

    return alias_map


def normalize_person_field(
    raw_value: str,
    alias_map: Dict[str, str],
    exclude_names: Optional[Set[str]] = None,
    is_bystander: bool = False,
) -> str:
    """
    Normaliseert een attribuutwaarde (S, A of bystanders) waarin één of meerdere
    personages gescheiden door '+' of ',' kunnen voorkomen.
    """
    if not raw_value:
        return "no" if is_bystander else "Unknown"

    val_stripped = raw_value.strip()
    if is_bystander and val_stripped.lower() in {"no", "none", "nobody", "n/a", "false", "", "yes"}:
        return "no" if val_stripped.lower() != "yes" else "yes"

    if not is_bystander and val_stripped.lower() in {"unknown", "?", "none", ""}:
        return "Unknown"

    parts = re.split(r"\s*\+\s*|\s*;\s*|\s*,\s*and\s+|\s+and\s+|\s*,\s*", val_stripped)
    normalized_list: List[str] = []
    seen: Set[str] = set()
    exclude = exclude_names or set()

    for part in parts:
        p = part.strip()
        if not p:
            continue
        p_low = p.lower()
        if is_bystander and p_low in {"no", "none", "nobody", "n/a"}:
            continue
        canon = alias_map.get(p_low, p)
        if canon in exclude:
            continue
        if canon not in seen:
            seen.add(canon)
            normalized_list.append(canon)

    if not normalized_list:
        return "no" if is_bystander else "Unknown"

    normalized_list.sort()
    return " + ".join(normalized_list)


# =====================================================================
# 6. Verificatie & Herstel van Geannoteerde Paragrafen (Geen Tekstverlies!)
# =====================================================================

ANY_TAG_RE = re.compile(r"</?(?:turn|interruption)\b[^>]*>", flags=re.IGNORECASE)


def strip_annotation_tags(xml_str: str) -> str:
    """Verwijdert enkel <turn> en <interruption> tags en decodeert XML-entities."""
    no_tags = ANY_TAG_RE.sub("", xml_str)
    return html.unescape(no_tags)


def verify_and_align_paragraph(original_text: str, annotated_xml: str) -> Optional[str]:
    """
    Controleert of `annotated_xml` geldig XML is en of de tekst na het verwijderen
    van <turn> en <interruption> tags exact overeenkomt met `original_text`.
    Indien de LLM kleine spaties of aanhalingstekens heeft gewijzigd, projecteert
    deze functie de tags exact terug op `original_text` (voorkomt mapping_failed!).
    """
    if not annotated_xml or "<turn" not in annotated_xml:
        return None

    fixed_amp = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)", "&amp;", annotated_xml)
    try:
        ET.fromstring(f"<p>{fixed_amp}</p>")
        annotated_xml = fixed_amp
    except ET.ParseError:
        return None

    plain_from_xml = strip_annotation_tags(annotated_xml)
    if plain_from_xml == original_text:
        return annotated_xml

    tokens = ANY_TAG_RE.split(annotated_xml)
    tags = ANY_TAG_RE.findall(annotated_xml)

    tag_positions: List[Tuple[int, str]] = []
    reconstructed = ""
    for i, tok in enumerate(tokens):
        reconstructed += html.unescape(tok)
        if i < len(tags):
            tag_positions.append((len(reconstructed), tags[i]))

    matcher = difflib.SequenceMatcher(None, reconstructed, original_text, autojunk=False)
    if matcher.ratio() < 0.90:
        return None

    pos_map: Dict[int, int] = {0: 0, len(reconstructed): len(original_text)}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size + 1):
            pos_map[block.a + offset] = block.b + offset

    sorted_keys = sorted(pos_map.keys())

    def map_pos(p: int) -> int:
        if p in pos_map:
            return pos_map[p]
        for k1, k2 in zip(sorted_keys, sorted_keys[1:]):
            if k1 <= p <= k2:
                if k2 == k1:
                    return pos_map[k1]
                ratio = (p - k1) / (k2 - k1)
                return int(round(pos_map[k1] + ratio * (pos_map[k2] - pos_map[k1])))
        return len(original_text)

    mapped_tags: Dict[int, List[str]] = {}
    for pos, tag in tag_positions:
        orig_pos = max(0, min(len(original_text), map_pos(pos)))
        mapped_tags.setdefault(orig_pos, []).append(tag)

    out_parts: List[str] = []
    for idx in range(len(original_text) + 1):
        if idx in mapped_tags:
            out_parts.extend(mapped_tags[idx])
        if idx < len(original_text):
            ch = original_text[idx]
            if ch == "&":
                out_parts.append("&amp;")
            elif ch == "<":
                out_parts.append("&lt;")
            elif ch == ">":
                out_parts.append("&gt;")
            else:
                out_parts.append(ch)

    realigned = "".join(out_parts)
    try:
        ET.fromstring(f"<p>{realigned}</p>")
        return realigned
    except ET.ParseError:
        return None


# =====================================================================
# 7. Gemini Batch-Annotatie met Sliding Context Window
# =====================================================================

SYSTEM_INSTRUCTION = """Je bent een computationeel-literair annotator gespecialiseerd in de romans van Jane Austen.
Je taak is om paragrafen uit een roman van Jane Austen te annoteren met <turn> en <interruption> XML-tags, exact volgens het schema van pride_and_prejudice_clean.xml, maar met twee belangrijke uitbreidingen:
1. Namen van sprekers (`S`), aangesprokenen (`A`) en omstanders (`bystanders`) moeten ALTIJD de genormaliseerde canonieke naam uit de gegeven personagelijst gebruiken (bv. 'Mrs. Elizabeth Bennet' in plaats van 'Lizzy' of 'Elizabeth').
2. Het attribuut `bystanders` bevat NIET enkel 'yes' of 'no', maar de EXACTE canonieke namen van alle andere personages die fysiek in de scène aanwezig zijn en het gesprek kunnen horen (gescheiden door ' + '), of 'no' indien er geen omstanders zijn (dus enkel spreker en aangesprokene aanwezig).

STRIKTE ANNOTATIEREGELS:
1. ONGEWIJZIGDE TEKST: Wijzig NOOIT een woord, letter of leesteken van de originele paragraaf. Voeg enkel `<turn ...>...</turn>` en `<interruption>...</interruption>` tags toe. Escape losse `&` in de tekst als `&amp;`.
2. DIRECTE REDE:
   - Omsluit elke spreekbeurt met `<turn S="..." A="..." bystanders="...">...</turn>`.
   - Als een spreekbeurt wordt onderbroken door een narratieve zegszin (zoals `said his lady to him one day,` of `replied Elizabeth,`), plaats die narratieve onderbreking dan BINNEN de `<turn>` in een `<interruption>...</interruption>` tag:
     Voorbeeld:
     `<turn S="Mrs. Bennet" A="Mr. Bennet" bystanders="no">"My dear Mr. Bennet," <interruption>said his lady to him one day,</interruption> "have you heard that Netherfield Park is let at last?"</turn>`
   - Als de narratieve zegszin helemaal NA of VOOR de directe rede staat (buiten de aanhalingstekens), laat die dan BUITEN de `<turn>` tag:
     Voorbeeld:
     `<turn S="Mrs. Bennet" A="Mr. Bennet" bystanders="no">"Do not you want to know who has taken it?"</turn> cried his wife impatiently.`
   - Als dezelfde spreker in dezelfde paragraaf van aangesprokene (`A`) verandert (bv. eerst tegen Mrs. Bennet spreekt en zich daarna tot Mary of Lydia richt), sluit dan de eerste `<turn>` af en open een nieuwe `<turn>` met de nieuwe `A` en bijgewerkte `bystanders`!
3. INDIRECTE REDE & VRIJE INDIRECTE REDE:
   - Annoteer duidelijke indirecte rede (bv. `Mr. Bennet replied that <turn S="Mr. Bennet" A="Mrs. Bennet" bystanders="no" indirectspeech="yes">he had not.</turn>`) met het extra attribuut `indirectspeech="yes"`.
   - Voeg `indirectspeech="yes"` ENKEL toe bij indirecte rede; laat het attribuut weg bij directe rede.
4. MEERDERE PERSONAGES:
   - Als een spreker zich tot meerdere personen tegelijk richt, scheid hun canonieke namen in `A` dan alfabetisch met ` + ` (bv. `A="Miss Catherine Bennet + Miss Lydia Bennet"`).
   - Als er andere personages in dezelfde ruimte/scène aanwezig zijn die niet de spreker (`S`) of aangesprokene (`A`) zijn, vermeld hun canonieke namen alfabetisch gescheiden door ` + ` in `bystanders` (bv. `bystanders="Miss Catherine Bennet + Miss Jane Bennet + Miss Lydia Bennet + Miss Mary Bennet + Mrs. Elizabeth Bennet"`).
   - Als er niemand anders aanwezig is buiten `S` en `A`, gebruik dan `bystanders="no"`.
"""


def annotate_batch(
    client: GeminiAnnotatorClient,
    canonical_characters: List[str],
    context_paragraphs: List[Tuple[int, str]],
    target_paragraphs: List[Tuple[int, str]],
    current_scene_characters: List[str],
    max_retries: int = 3,
) -> Tuple[Dict[int, str], List[str]]:
    """
    Annoteert een batch van doelparagrafen met behulp van Gemini en geeft een
    dictionary {paragraph_idx: annotated_xml} en de geüpdatete lijst van aanwezige
    scène-personages terug.
    """
    char_list_str = ", ".join(sorted(set(canonical_characters)))
    context_str = "\n".join([f"[Context p={idx}] {text}" for idx, text in context_paragraphs])
    target_str = "\n".join([f"[Target p={idx}] {text}" for idx, text in target_paragraphs])
    scene_str = ", ".join(current_scene_characters) if current_scene_characters else "Nog te bepalen uit de tekst"

    prompt = f"""CANONIEKE PERSONAGELIJST (gebruik deze exacte namen voor S, A en bystanders):
{char_list_str}

HUIDIGE AANWEZIGE PERSONAGES AAN HET BEGIN VAN DEZE BATCH:
{scene_str}

VOORAFGAANDE CONTEXT (enkel ter referentie voor wie in de ruimte is en wie spreekt):
{context_str}

TE ANNOTEREN DOELPARAGRAFEN (geef voor ELKE Target p=idx een item terug in `paragraphs`):
{target_str}
"""

    for attempt in range(max_retries):
        try:
            data = client.generate_structured(
                prompt=prompt,
                response_schema=BATCH_ANNOTATION_SCHEMA,
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.1,
            )
            paragraphs_out = data.get("paragraphs", [])
            if not paragraphs_out:
                raise ValueError("Lege respons van model.")

            annotated_map: Dict[int, str] = {}
            latest_scene = current_scene_characters
            orig_lookup = dict(target_paragraphs)

            for item in paragraphs_out:
                p_idx = int(item.get("paragraph_index", -1))
                latest_scene = item.get("scene_characters_present") or latest_scene
                has_speech = bool(item.get("has_speech", False))
                ann_xml = item.get("annotated_xml", "")
                if has_speech and ann_xml and p_idx in orig_lookup:
                    aligned = verify_and_align_paragraph(orig_lookup[p_idx], ann_xml)
                    if aligned:
                        annotated_map[p_idx] = aligned

            return annotated_map, latest_scene

        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"  [Poging {attempt + 1}/{max_retries} mislukt: {e}] Wacht {wait}s...", file=sys.stderr)
            time.sleep(wait)

    return {}, current_scene_characters


# =====================================================================
# 8. Post-processing: Globale Naamnormalisatie & Volgnummering (n="1"..)
# =====================================================================

def resolve_unknown_aliases_with_llm(
    client: GeminiAnnotatorClient,
    title: str,
    unknown_names: List[str],
    alias_map: Dict[str, str],
) -> None:
    """
    Vraagt Gemini na afloop om eventuele overgebleven niet-canonieke namen
    alsnog te koppelen aan de juiste canonieke naam.
    """
    if not unknown_names:
        return
    known_canonicals = sorted(set(alias_map.values()))
    prompt = f"""In onze XML-annotatie van "{title}" van Jane Austen kwamen nog de volgende
ruwe personage-aanduidingen voor:
{json.dumps(unknown_names, ensure_ascii=False)}

Bestaande canonieke namen:
{json.dumps(known_canonicals, ensure_ascii=False)}

Koppel elke ruwe naam aan de juiste canonieke naam (of maak een nette formele canonieke naam
als het een nieuw bijpersonage betreft, bv. 'Mrs. Elizabeth Bennet' voor Lizzy/Elizabeth).
"""
    try:
        data = client.generate_structured(
            prompt=prompt,
            response_schema=ALIAS_RESOLUTION_SCHEMA,
            temperature=0.0,
        )
        for item in data.get("resolutions", []):
            raw_n = item.get("raw_name", "").lower().strip()
            can_n = item.get("canonical_name", "").strip()
            if raw_n and can_n:
                alias_map[raw_n] = can_n
    except Exception as e:
        print(f"[Waarschuwing] Kon extra aliassen niet resolven: {e}", file=sys.stderr)


def build_final_xml(
    title: str,
    author: str,
    model_name: str,
    paragraphs: List[str],
    annotated_paragraphs: Dict[int, str],
    alias_map: Dict[str, str],
    client: Optional[GeminiAnnotatorClient] = None,
) -> ET.ElementTree:
    """
    Bouwt de definitieve XML-boom, verzamelt eventuele onbekende namen voor een laatste
    LLM-normalisatieronde, normaliseert S, A en bystanders, en nummert alle turns (n="1", ...).
    """
    root = ET.Element("book")
    root.set("title", title)
    root.set("author", author)
    root.set("model", model_name)

    raw_names_found: Set[str] = set()
    for idx, orig_p in enumerate(paragraphs):
        if idx in annotated_paragraphs:
            xml_p_str = f"<p>{annotated_paragraphs[idx]}</p>"
            try:
                p_elem = ET.fromstring(xml_p_str)
                for turn in p_elem.findall(".//turn"):
                    for attr in ("S", "A", "bystanders"):
                        val = turn.get(attr, "")
                        for part in re.split(r"\s*\+\s*", val):
                            p_clean = part.strip()
                            if p_clean and p_clean.lower() not in {"no", "yes", "unknown", "?"}:
                                if p_clean.lower() not in alias_map:
                                    raw_names_found.add(p_clean)
                root.append(p_elem)
                continue
            except ET.ParseError:
                pass

        p_elem = ET.Element("p")
        p_elem.text = orig_p
        root.append(p_elem)

    if client and raw_names_found:
        resolve_unknown_aliases_with_llm(client, title, sorted(raw_names_found), alias_map)

    turn_counter = 1
    for p_elem in root.findall("p"):
        for turn in p_elem.findall(".//turn"):
            raw_s = turn.get("S", "Unknown")
            raw_a = turn.get("A", "Unknown")
            raw_b = turn.get("bystanders", "no")
            indirect = turn.get("indirectspeech", None)

            norm_s = normalize_person_field(raw_s, alias_map, is_bystander=False)
            norm_a = normalize_person_field(raw_a, alias_map, is_bystander=False)

            active_participants = set(norm_s.split(" + ")) | set(norm_a.split(" + "))
            norm_b = normalize_person_field(
                raw_b, alias_map, exclude_names=active_participants, is_bystander=True
            )

            turn.attrib.clear()
            turn.set("n", str(turn_counter))
            turn.set("S", norm_s)
            turn.set("A", norm_a)
            turn.set("bystanders", norm_b)
            if indirect and indirect.lower() in {"yes", "true", "x", "xfid"}:
                turn.set("indirectspeech", "yes")

            turn_counter += 1

    return ET.ElementTree(root)


# =====================================================================
# 9. CLI Entrypoint (1 verplichte parameter: het .txt bestand)
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annoteer een boek van Jane Austen (.txt) naar een genormaliseerde XML."
    )
    parser.add_argument(
        "input_txt",
        type=Path,
        help="Pad naar het input .txt bestand van het boek (bv. pride_and_prejudice.txt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optioneel pad voor het output XML-bestand (standaard: <input_stem>_annotated.xml)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gemini-2.5-pro",
        help="Gemini modelnaam (standaard: gemini-2.5-pro, of gemini-2.5-flash voor snelheid)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Aantal paragrafen per API-call (standaard: 20)",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=8,
        help="Aantal voorafgaande paragrafen als context-window (standaard: 8)",
    )
    args = parser.parse_args()

    if not args.input_txt.exists():
        print(f"Fout: Bestand niet gevonden: {args.input_txt}", file=sys.stderr)
        sys.exit(1)

    output_xml = args.output or args.input_txt.with_name(f"{args.input_txt.stem}_annotated.xml")
    checkpoint_file = args.input_txt.with_name(f"{args.input_txt.stem}_checkpoint.json")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print(
            "Fout: Stel eerst de omgevingsvariabele GEMINI_API_KEY in:\n"
            "  export GEMINI_API_KEY='jouw-api-key'",
            file=sys.stderr,
        )
        sys.exit(1)

    client = GeminiAnnotatorClient(api_key=api_key, model_name=args.model)

    print(f"[1/4] Inlezen en splitsen van '{args.input_txt}'...")
    title, author, paragraphs, start_idx, end_idx = parse_gutenberg_text(args.input_txt)
    print(
        f"      Titel: {title} | Auteur: {author}\n"
        f"      Totaal paragrafen: {len(paragraphs)} (Literaire tekst: {start_idx} t/m {end_idx - 1})"
    )

    print(f"[2/4] Opbouwen van canoniek personageregister met {args.model}...")
    literary_paragraphs = paragraphs[start_idx:end_idx]
    alias_map = build_character_registry(client, title, author, literary_paragraphs)
    canonical_characters = sorted(set(alias_map.values()))
    print(f"      {len(canonical_characters)} canonieke personages geregistreerd.")

    annotated_paragraphs: Dict[int, str] = {}
    last_processed_idx = start_idx - 1
    current_scene_characters: List[str] = []

    if checkpoint_file.exists():
        try:
            ckpt = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            annotated_paragraphs = {int(k): v for k, v in ckpt.get("annotated_paragraphs", {}).items()}
            last_processed_idx = ckpt.get("last_processed_idx", start_idx - 1)
            current_scene_characters = ckpt.get("current_scene_characters", [])
            print(f"      Checkpoint hervat vanaf paragraaf {last_processed_idx + 1}.")
        except Exception:
            pass

    print(f"[3/4] Annoteren van paragrafen in batches van {args.batch_size}...")
    for batch_start in range(max(start_idx, last_processed_idx + 1), end_idx, args.batch_size):
        batch_end = min(end_idx, batch_start + args.batch_size)
        ctx_start = max(start_idx, batch_start - args.context_size)

        context_batch = [(i, paragraphs[i]) for i in range(ctx_start, batch_start)]
        target_batch = [(i, paragraphs[i]) for i in range(batch_start, batch_end)]

        print(f"      Verwerken van paragraaf {batch_start}..{batch_end - 1} / {end_idx - 1}...")
        batch_result, current_scene_characters = annotate_batch(
            client=client,
            canonical_characters=canonical_characters,
            context_paragraphs=context_batch,
            target_paragraphs=target_batch,
            current_scene_characters=current_scene_characters,
        )
        annotated_paragraphs.update(batch_result)

        checkpoint_data = {
            "last_processed_idx": batch_end - 1,
            "current_scene_characters": current_scene_characters,
            "annotated_paragraphs": annotated_paragraphs,
        }
        checkpoint_file.write_text(json.dumps(checkpoint_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[4/4] Normaliseren van namen en wegschrijven van XML naar '{output_xml}'...")
    tree = build_final_xml(
        title=title,
        author=author,
        model_name=args.model,
        paragraphs=paragraphs,
        annotated_paragraphs=annotated_paragraphs,
        alias_map=alias_map,
        client=client,
    )
    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    print(f"Klaar! Geannoteerde XML opgeslagen in: {output_xml}")


if __name__ == "__main__":
    main()

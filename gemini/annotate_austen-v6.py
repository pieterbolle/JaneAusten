#!/usr/bin/env python3
"""
annotate_austen.py (v6)
-----------------------
Annotates a novel by Jane Austen (.txt) into a normalized XML file, including:
  - Detection of direct speech, indirect speech (indirectspeech="IS"), and
    free indirect discourse (indirectspeech="FID").
  - Explicit annotation of letters/notes and soliloquies (speaking to oneself).
  - Identification of <interruption> tags within a <turn>.
  - Explicit listing of present bystanders (separated by " + ", or "no" if none).
  - Pure, unblended confidence scoring (`confidence="0.00..1.00"`) on `<p>`, `<turn>`,
    and `<interruption>` for downstream researcher cutoff analysis (without premature
    `needs_review` attributes).
  - Strict opcode-level text verification (forbidding alphanumeric edits), mid-word tag
    boundary rejection, mandatory attribute validation (`S`, `A`, `bystanders`), strict
    quote & `has_speech` completeness across all attempts, full checkpoint metadata
    verification, cascading scene-state propagation across batch boundaries, and fail-fast
    XML export verification.

Usage:
    python annotate_austen-v6.py pride_and_prejudice.txt
    python annotate_austen-v6.py pride_and_prejudice.txt --upgrade-is-fid
    python annotate_austen-v6.py pride_and_prejudice.txt --fresh
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Optional import of the official google-genai SDK (automatically falls back to
# standard library urllib.request if google-genai is not installed and an API key is used).
try:
    from google import genai
    from google.genai import types
    HAS_GENAI_SDK = True
except ImportError:
    HAS_GENAI_SDK = False

SCHEMA_VERSION = "6.0"


# =====================================================================
# 1. JSON Schemas for Gemini Structured Outputs
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
                            "The normalized, canonical name of the character throughout the entire book, "
                            "e.g., 'Mrs. Elizabeth Bennet', 'Mr. Fitzwilliam Darcy', 'Miss Emma Woodhouse'."
                        ),
                    },
                    "aliases": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": (
                            "All possible first names, nicknames, titles, maiden names, married names, and "
                            "descriptive references occurring in the text."
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
                        "description": "The exact integer index (p=...) of the paragraph from the input batch.",
                    },
                    "scene_characters_present": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": (
                            "List of all characters (using their canonical name) physically present "
                            "in the current scene/room at the end of this paragraph. Use [] if nobody is present."
                        ),
                    },
                    "has_speech": {
                        "type": "BOOLEAN",
                        "description": (
                            "True if this paragraph contains direct speech, letters, soliloquy, indirect speech (IS), "
                            "or free indirect discourse (FID) that should be annotated with <turn>."
                        ),
                    },
                    "paragraph_confidence": {
                        "type": "NUMBER",
                        "description": (
                            "Confidence score between 0.0 and 1.0 for the overall annotation decision of this "
                            "paragraph (including whether has_speech is True/False and who is present in the scene)."
                        ),
                    },
                    "annotated_xml": {
                        "type": "STRING",
                        "description": (
                            "If has_speech=True: the EXACT text of the paragraph (without changing a single "
                            'word or punctuation mark) containing <turn ... confidence="0.XX"> and optional '
                            '<interruption confidence="0.XX"> tags. If has_speech=False: leave empty ("").'
                        ),
                    },
                },
                "required": [
                    "paragraph_index",
                    "scene_characters_present",
                    "has_speech",
                    "paragraph_confidence",
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
# 2. Comprehensive DEFAULT_AUSTEN_NORMALIZATION for all 7 novels
# =====================================================================

DEFAULT_AUSTEN_NORMALIZATION: Dict[str, Dict[str, str]] = {
    "pride and prejudice": {
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
        "jane": "Miss Jane Bennet",
        "miss bennet": "Miss Jane Bennet",
        "jane bennet": "Miss Jane Bennet",
        "miss jane bennet": "Miss Jane Bennet",
        "mrs. bingley": "Miss Jane Bennet",
        "the eldest miss bennet": "Miss Jane Bennet",
        "kitty": "Miss Catherine Bennet",
        "catherine": "Miss Catherine Bennet",
        "miss catherine bennet": "Miss Catherine Bennet",
        "kitty bennet": "Miss Catherine Bennet",
        "catherine bennet": "Miss Catherine Bennet",
        "lydia": "Miss Lydia Bennet",
        "miss lydia": "Miss Lydia Bennet",
        "miss lydia bennet": "Miss Lydia Bennet",
        "lydia bennet": "Miss Lydia Bennet",
        "mrs. wickham": "Miss Lydia Bennet",
        "mary": "Miss Mary Bennet",
        "miss mary bennet": "Miss Mary Bennet",
        "mary bennet": "Miss Mary Bennet",
        "mr. bennet": "Mr. Bennet",
        "mrs. bennet": "Mrs. Bennet",
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
        "mr. wickham": "Mr. George Wickham",
        "wickham": "Mr. George Wickham",
        "george wickham": "Mr. George Wickham",
        "mr. george wickham": "Mr. George Wickham",
        "mr. denny": "Mr. Denny",
        "denny": "Mr. Denny",
        "colonel forster": "Colonel Forster",
        "mrs. forster": "Mrs. Forster",
        "harriet": "Mrs. Forster",
        "mr. gardiner": "Mr. Edward Gardiner",
        "edw. gardiner": "Mr. Edward Gardiner",
        "mr. edward gardiner": "Mr. Edward Gardiner",
        "mrs. gardiner": "Mrs. Gardiner",
        "m. gardiner": "Mrs. Gardiner",
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
        "mr. jones": "Mr. Jones",
        "mrs. long": "Mrs. Long",
        "captain carter": "Captain Carter",
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
    "emma": {
        "emma": "Miss Emma Woodhouse",
        "emma woodhouse": "Miss Emma Woodhouse",
        "miss woodhouse": "Miss Emma Woodhouse",
        "miss emma woodhouse": "Miss Emma Woodhouse",
        "mr. woodhouse": "Mr. Henry Woodhouse",
        "henry woodhouse": "Mr. Henry Woodhouse",
        "mr. knightley": "Mr. George Knightley",
        "knightley": "Mr. George Knightley",
        "george knightley": "Mr. George Knightley",
        "mr. george knightley": "Mr. George Knightley",
        "mr. john knightley": "Mr. John Knightley",
        "john knightley": "Mr. John Knightley",
        "isabella": "Mrs. Isabella Knightley",
        "mrs. john knightley": "Mrs. Isabella Knightley",
        "mrs. isabella knightley": "Mrs. Isabella Knightley",
        "harriet": "Miss Harriet Smith",
        "harriet smith": "Miss Harriet Smith",
        "miss smith": "Miss Harriet Smith",
        "miss harriet smith": "Miss Harriet Smith",
        "frank": "Mr. Frank Churchill",
        "frank churchill": "Mr. Frank Churchill",
        "mr. frank churchill": "Mr. Frank Churchill",
        "mr. churchill": "Mr. Frank Churchill",
        "jane": "Miss Jane Fairfax",
        "jane fairfax": "Miss Jane Fairfax",
        "miss fairfax": "Miss Jane Fairfax",
        "miss jane fairfax": "Miss Jane Fairfax",
        "mr. elton": "Mr. Philip Elton",
        "elton": "Mr. Philip Elton",
        "philip elton": "Mr. Philip Elton",
        "mrs. elton": "Mrs. Augusta Elton",
        "augusta": "Mrs. Augusta Elton",
        "miss hawkins": "Mrs. Augusta Elton",
        "augusta hawkins": "Mrs. Augusta Elton",
        "mrs. weston": "Mrs. Anne Weston",
        "miss taylor": "Mrs. Anne Weston",
        "mr. weston": "Mr. Weston",
        "weston": "Mr. Weston",
        "miss bates": "Miss Bates",
        "mrs. bates": "Mrs. Bates",
        "robert martin": "Mr. Robert Martin",
        "mr. martin": "Mr. Robert Martin",
        "elizabeth martin": "Miss Elizabeth Martin",
        "miss martin": "Miss Elizabeth Martin",
        "mr. perry": "Mr. Perry",
        "perry": "Mr. Perry",
        "mr. cole": "Mr. Cole",
        "mrs. cole": "Mrs. Cole",
        "mrs. goddard": "Mrs. Goddard",
    },
    "lady susan": {
        "lady susan": "Lady Susan Vernon",
        "lady susan vernon": "Lady Susan Vernon",
        "susan vernon": "Lady Susan Vernon",
        "frederica": "Miss Frederica Vernon",
        "frederica vernon": "Miss Frederica Vernon",
        "miss vernon": "Miss Frederica Vernon",
        "miss frederica vernon": "Miss Frederica Vernon",
        "catherine": "Mrs. Catherine Vernon",
        "catherine vernon": "Mrs. Catherine Vernon",
        "mrs. vernon": "Mrs. Catherine Vernon",
        "mrs. catherine vernon": "Mrs. Catherine Vernon",
        "charles": "Mr. Charles Vernon",
        "charles vernon": "Mr. Charles Vernon",
        "mr. vernon": "Mr. Charles Vernon",
        "mr. charles vernon": "Mr. Charles Vernon",
        "reginald": "Mr. Reginald De Courcy",
        "reginald de courcy": "Mr. Reginald De Courcy",
        "mr. de courcy": "Mr. Reginald De Courcy",
        "mr. reginald de courcy": "Mr. Reginald De Courcy",
        "sir reginald": "Sir Reginald De Courcy",
        "sir reginald de courcy": "Sir Reginald De Courcy",
        "lady de courcy": "Lady De Courcy",
        "alicia": "Mrs. Alicia Johnson",
        "alicia johnson": "Mrs. Alicia Johnson",
        "mrs. johnson": "Mrs. Alicia Johnson",
        "mr. johnson": "Mr. Johnson",
        "sir james": "Sir James Martin",
        "sir james martin": "Sir James Martin",
        "mr. manwaring": "Mr. Manwaring",
        "manwaring": "Mr. Manwaring",
        "mrs. manwaring": "Mrs. Manwaring",
        "miss manwaring": "Miss Maria Manwaring",
        "maria manwaring": "Miss Maria Manwaring",
    },
    "mansfield park": {
        "fanny": "Miss Fanny Price",
        "fanny price": "Miss Fanny Price",
        "miss price": "Miss Fanny Price",
        "miss fanny price": "Miss Fanny Price",
        "edmund": "Mr. Edmund Bertram",
        "edmund bertram": "Mr. Edmund Bertram",
        "mr. edmund bertram": "Mr. Edmund Bertram",
        "sir thomas": "Sir Thomas Bertram",
        "sir thomas bertram": "Sir Thomas Bertram",
        "lady bertram": "Lady Maria Bertram",
        "mrs. norris": "Mrs. Norris",
        "aunt norris": "Mrs. Norris",
        "tom": "Mr. Tom Bertram",
        "tom bertram": "Mr. Tom Bertram",
        "mr. bertram": "Mr. Tom Bertram",
        "mr. tom bertram": "Mr. Tom Bertram",
        "maria": "Miss Maria Bertram",
        "maria bertram": "Miss Maria Bertram",
        "miss bertram": "Miss Maria Bertram",
        "mrs. rushworth": "Miss Maria Bertram",
        "julia": "Miss Julia Bertram",
        "julia bertram": "Miss Julia Bertram",
        "miss julia bertram": "Miss Julia Bertram",
        "henry": "Mr. Henry Crawford",
        "henry crawford": "Mr. Henry Crawford",
        "mr. crawford": "Mr. Henry Crawford",
        "mr. henry crawford": "Mr. Henry Crawford",
        "crawford": "Mr. Henry Crawford",
        "mary": "Miss Mary Crawford",
        "mary crawford": "Miss Mary Crawford",
        "miss crawford": "Miss Mary Crawford",
        "miss mary crawford": "Miss Mary Crawford",
        "mr. rushworth": "Mr. James Rushworth",
        "rushworth": "Mr. James Rushworth",
        "james rushworth": "Mr. James Rushworth",
        "mrs. rushworth senior": "Mrs. Rushworth",
        "mr. yates": "Mr. John Yates",
        "yates": "Mr. John Yates",
        "william": "Mr. William Price",
        "william price": "Mr. William Price",
        "susan": "Miss Susan Price",
        "susan price": "Miss Susan Price",
        "mrs. price": "Mrs. Frances Price",
        "mr. price": "Mr. Price",
        "dr. grant": "Dr. Grant",
        "mrs. grant": "Mrs. Grant",
    },
    "northanger abbey": {
        "catherine": "Miss Catherine Morland",
        "catherine morland": "Miss Catherine Morland",
        "miss morland": "Miss Catherine Morland",
        "miss catherine morland": "Miss Catherine Morland",
        "henry": "Mr. Henry Tilney",
        "henry tilney": "Mr. Henry Tilney",
        "mr. tilney": "Mr. Henry Tilney",
        "mr. henry tilney": "Mr. Henry Tilney",
        "eleanor": "Miss Eleanor Tilney",
        "eleanor tilney": "Miss Eleanor Tilney",
        "miss tilney": "Miss Eleanor Tilney",
        "miss eleanor tilney": "Miss Eleanor Tilney",
        "general tilney": "General Tilney",
        "the general": "General Tilney",
        "captain tilney": "Captain Frederick Tilney",
        "frederick": "Captain Frederick Tilney",
        "frederick tilney": "Captain Frederick Tilney",
        "isabella": "Miss Isabella Thorpe",
        "isabella thorpe": "Miss Isabella Thorpe",
        "miss thorpe": "Miss Isabella Thorpe",
        "miss isabella thorpe": "Miss Isabella Thorpe",
        "john": "Mr. John Thorpe",
        "john thorpe": "Mr. John Thorpe",
        "mr. thorpe": "Mr. John Thorpe",
        "thorpe": "Mr. John Thorpe",
        "mrs. thorpe": "Mrs. Thorpe",
        "james": "Mr. James Morland",
        "james morland": "Mr. James Morland",
        "mr. morland": "Mr. James Morland",
        "mrs. allen": "Mrs. Allen",
        "mr. allen": "Mr. Allen",
        "mrs. morland": "Mrs. Morland",
    },
    "persuasion": {
        "anne": "Miss Anne Elliot",
        "anne elliot": "Miss Anne Elliot",
        "miss anne": "Miss Anne Elliot",
        "miss anne elliot": "Miss Anne Elliot",
        "wentworth": "Captain Frederick Wentworth",
        "captain wentworth": "Captain Frederick Wentworth",
        "frederick wentworth": "Captain Frederick Wentworth",
        "frederick": "Captain Frederick Wentworth",
        "sir walter": "Sir Walter Elliot",
        "sir walter elliot": "Sir Walter Elliot",
        "elizabeth": "Miss Elizabeth Elliot",
        "elizabeth elliot": "Miss Elizabeth Elliot",
        "miss elliot": "Miss Elizabeth Elliot",
        "mary": "Mrs. Mary Musgrove",
        "mary musgrove": "Mrs. Mary Musgrove",
        "mrs. charles musgrove": "Mrs. Mary Musgrove",
        "charles": "Mr. Charles Musgrove",
        "charles musgrove": "Mr. Charles Musgrove",
        "mr. charles musgrove": "Mr. Charles Musgrove",
        "lady russell": "Lady Russell",
        "mr. elliot": "Mr. William Elliot",
        "william elliot": "Mr. William Elliot",
        "mr. william elliot": "Mr. William Elliot",
        "louisa": "Miss Louisa Musgrove",
        "louisa musgrove": "Miss Louisa Musgrove",
        "miss louisa musgrove": "Miss Louisa Musgrove",
        "henrietta": "Miss Henrietta Musgrove",
        "henrietta musgrove": "Miss Henrietta Musgrove",
        "miss musgrove": "Miss Henrietta Musgrove",
        "mr. musgrove": "Mr. Musgrove",
        "mrs. musgrove": "Mrs. Musgrove",
        "admiral croft": "Admiral Croft",
        "the admiral": "Admiral Croft",
        "mrs. croft": "Mrs. Sophia Croft",
        "sophia": "Mrs. Sophia Croft",
        "captain benwick": "Captain James Benwick",
        "benwick": "Captain James Benwick",
        "james benwick": "Captain James Benwick",
        "captain harville": "Captain Harville",
        "harville": "Captain Harville",
        "mrs. harville": "Mrs. Harville",
        "mrs. clay": "Mrs. Penelope Clay",
        "penelope clay": "Mrs. Penelope Clay",
        "mrs. smith": "Mrs. Smith",
        "nurse rook": "Nurse Rooke",
        "nurse rooke": "Nurse Rooke",
        "lady dalrymple": "Lady Dalrymple",
        "miss carteret": "Miss Carteret",
    },
}


# =====================================================================
# 3. Gemini API Wrapper (supports Vertex AI ADC & AI Studio API Key)
# =====================================================================

class GeminiAnnotatorClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        location: str = "eu",
        model_name: str = "gemini-4-argon",
    ) -> None:
        self.api_key = api_key
        self.project_id = project_id
        self.location = location
        self.model_name = model_name

        if HAS_GENAI_SDK:
            if project_id:
                self._sdk_client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=location,
                )
            else:
                self._sdk_client = genai.Client(api_key=api_key)
        else:
            if project_id and not api_key:
                raise RuntimeError(
                    "To use Vertex AI (GOOGLE_CLOUD_PROJECT), please install the official SDK:\n"
                    "  pip install -r requirements.txt"
                )
            self._sdk_client = None

    def generate_structured(
        self,
        prompt: str,
        response_schema: Dict[str, Any],
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Calls the Gemini model and returns the parsed JSON response."""
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
        cand = body["candidates"][0]
        text_out = cand["content"]["parts"][0]["text"]
        return json.loads(text_out)


# =====================================================================
# 4. Helper Functions for Text, Boilerplate & Atomic Checkpointing
# =====================================================================

def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of the input file for checkpoint integrity tracking."""
    h = hashlib.sha256()
    h.update(filepath.read_bytes())
    return h.hexdigest()


def atomic_write_json(filepath: Path, data: Dict[str, Any]) -> None:
    """Writes JSON data atomically via a temporary file and os.replace."""
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, filepath)


def parse_gutenberg_text(filepath: Path) -> Tuple[str, str, List[str], int, int]:
    """
    Reads a Project Gutenberg .txt file with strict UTF-8 decoding (falling back
    with an explicit warning only if non-UTF-8 bytes exist), extracts title and author,
    splits the text into paragraphs, and locates the start and end indices of the novel body.
    """
    raw_bytes = filepath.read_bytes()
    try:
        raw_text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        print(
            f"[Warning] Input file '{filepath}' contains non-UTF-8 bytes ({exc}); decoding with latin-1 fallback.",
            file=sys.stderr,
        )
        raw_text = raw_bytes.decode("latin-1")

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

    # Universal chapter/letter opener detection across all 7 Austen novels (skipping Table of Contents)
    for i in range(start_idx, min(start_idx + 120, end_idx)):
        p_str = paragraphs[i].strip()
        if re.match(r"^(CHAPTER\s+(I|1)\b|LETTER\s+(I|1)\b|I\.\s*$|I\.\s+[A-Z])", p_str, re.IGNORECASE) and len(p_str) < 90:
            next_few = [paragraphs[j].strip() for j in range(i + 1, min(i + 4, end_idx))]
            if sum(1 for nf in next_few if re.match(r"^(CHAPTER|LETTER)\b", nf, re.IGNORECASE)) >= 2:
                continue
            start_idx = i + 1 if len(p_str) < 30 else i
            break

    return title, author, paragraphs, start_idx, end_idx


def is_boilerplate_paragraph(text: str, book_title: str = "") -> bool:
    """
    Universal boilerplate detector for all 7 Jane Austen novels (headers, volume markers,
    title pages, table of contents lines, and illustration tags).
    """
    t_strip = text.strip()
    t_low = t_strip.lower()
    title_clean = book_title.lower().strip()

    if len(t_strip) < 120:
        if title_clean and (t_low == title_clean or t_low.rstrip(".:;") == title_clean):
            return True
        for known_title in DEFAULT_AUSTEN_NORMALIZATION:
            if t_low.rstrip(".:;") == known_title or t_low.startswith(f"{known_title}:"):
                return True
        if (
            "project gutenberg" in t_low
            or "by the author of" in t_low
            or "in three volumes" in t_low
            or "in two volumes" in t_low
            or "transcriber's note" in t_low
        ):
            return True

    if re.match(
        r"^(CHAPTER\s+[IVXLCDM0-9]+|VOLUME\s+[IVXLCDM0-9]+|VOL\.|END OF VOL|LETTER\s+[IVXLCDM0-9]+|A Novel|London:|\[Illustration|CONTENTS\b)",
        t_strip,
        re.IGNORECASE,
    ):
        return True
    return False


# =====================================================================
# 5. Building the Canonical Character Registry & Name Normalization
# =====================================================================

def build_character_registry(
    client: GeminiAnnotatorClient,
    title: str,
    author: str,
    sample_paragraphs: List[str],
) -> Dict[str, str]:
    """Builds an alias -> canonical_name mapping for the entire book."""
    alias_map: Dict[str, str] = {}

    title_key = title.lower().strip()
    for known_title, mapping in DEFAULT_AUSTEN_NORMALIZATION.items():
        if known_title in title_key or known_title.replace(" ", "_") in title_key:
            for k, v in mapping.items():
                alias_map[k.lower().strip()] = v
                alias_map[v.lower().strip()] = v

    n_total = len(sample_paragraphs)
    if n_total <= 180:
        sampled = sample_paragraphs
    else:
        sampled = (
            sample_paragraphs[:80]
            + sample_paragraphs[n_total // 2 - 25 : n_total // 2 + 25]
            + sample_paragraphs[-50:]
        )
    sample_text = "\n".join(sampled)[:28000]

    prompt = f"""You are a literary scholar specializing in the works of {author}.
We are annotating the novel "{title}" by {author}.
Create a comprehensive list of ALL characters in "{title}" (both major and minor characters).

Provide for each character:
1. `canonical_name`: A single unique, formal normalized name (e.g., 'Mrs. Elizabeth Bennet' for Elizabeth/Lizzy/Eliza).
2. `aliases`: All possible name variants, first names, nicknames, surnames, titles before and after marriage, and relational references.

Excerpts from the text for reference:
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
        print(
            f"[Warning] Could not fetch dynamic character registry ({e}); using built-in defaults.",
            file=sys.stderr,
        )

    return alias_map


def validate_turn_attributes_strict(turn_elem: ET.Element) -> None:
    """
    Validates that a <turn> element returned by the model has explicit, non-empty
    `S`, `A`, and `bystanders` attributes, that `bystanders` is not 'yes' or empty,
    and that `indirectspeech` (if present) is strictly 'IS' or 'FID'.
    """
    raw_s = turn_elem.get("S")
    raw_a = turn_elem.get("A")
    raw_b = turn_elem.get("bystanders")

    if raw_s is None or not raw_s.strip():
        raise ValueError("Missing or empty 'S' (Speaker) attribute on <turn>.")
    if raw_a is None or not raw_a.strip():
        raise ValueError("Missing or empty 'A' (Addressee) attribute on <turn>.")
    if raw_b is None or not raw_b.strip():
        raise ValueError("Missing or empty 'bystanders' attribute on <turn>.")

    b_low = raw_b.strip().lower()
    if b_low in {"yes", "true", "unknown", "?"}:
        raise ValueError(
            f"Invalid bystanders='{raw_b}' on <turn>: must be 'no' or explicit canonical character names."
        )

    is_attr = turn_elem.get("indirectspeech")
    if is_attr is not None:
        normalize_indirect_speech_attr(is_attr, strict=True)


def normalize_person_field(
    raw_value: Optional[str],
    alias_map: Dict[str, str],
    exclude_names: Optional[Set[str]] = None,
    is_bystander: bool = False,
) -> str:
    """
    Normalizes an attribute value (S, A, or bystanders) that may contain one or
    more characters separated by '+' or ','.
    Raises ValueError if the input is missing/empty or if bystanders='yes'.
    """
    if raw_value is None or not raw_value.strip():
        raise ValueError(f"Missing or empty {'bystanders' if is_bystander else 'S/A'} attribute.")

    val_stripped = raw_value.strip()
    val_low = val_stripped.lower()

    if is_bystander:
        if val_low in {"yes", "true", "unknown", "?"}:
            raise ValueError(f"Invalid bystanders value '{raw_value}': must be 'no' or explicit names.")
        if val_low in {"no", "none", "nobody", "n/a", "false"}:
            return "no"
    else:
        if val_low in {"unknown", "?", "none"}:
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


def normalize_indirect_speech_attr(raw_val: Optional[str], strict: bool = True) -> Optional[str]:
    """
    Normalizes the indirectspeech attribute to 'IS' (Indirect Speech) or
    'FID' (Free Indirect Discourse), or None for direct speech.
    Raises ValueError on unrecognized non-empty labels when `strict=True`.
    """
    if not raw_val:
        return None
    v = raw_val.strip().upper()
    if v in {"NO", "FALSE", "NONE", "DS", "DIRECT", ""}:
        return None
    if v in {"FID", "XFID", "FREE INDIRECT", "FREE_INDIRECT", "FREE INDIRECT DISCOURSE"}:
        return "FID"
    if v in {"IS", "YES", "TRUE", "X", "INDIRECT", "INDIRECT SPEECH", "INDIRECT_SPEECH"}:
        return "IS"
    if strict:
        raise ValueError(f"Unrecognized indirectspeech attribute value: '{raw_val}' (expected 'IS' or 'FID').")
    return "IS"


def format_confidence(val: Any, default: float = 0.90) -> Tuple[float, str]:
    """Clamps a confidence value to [0.0, 1.0] and returns (raw_float, formatted_2dec_str)."""
    try:
        f = float(val)
        if math.isnan(f):
            f = default
    except (TypeError, ValueError):
        f = default
    f = max(0.0, min(1.0, f))
    return f, f"{f:.2f}"


# =====================================================================
# 6. Strict Opcode & Boundary Verification of Annotated Paragraphs
# =====================================================================

ANY_TAG_RE = re.compile(r"</?(?:turn|interruption)\b[^>]*>", flags=re.IGNORECASE)
ALLOWED_DIFF_CHARS_RE = re.compile(r'^[\s"\'\u2018\u2019\u201c\u201d\u2014\u2013\-]+$')


def strip_annotation_tags(xml_str: str) -> str:
    """Removes only <turn> and <interruption> tags and unescapes XML entities."""
    no_tags = ANY_TAG_RE.sub("", xml_str)
    return html.unescape(no_tags)


def check_tag_boundaries_not_midword(plain_text: str, tag_positions: List[Tuple[int, str]]) -> bool:
    """
    Verifies that no <turn> or <interruption> tag is inserted in the middle of an
    alphanumeric word (i.e., where both plain_text[pos - 1] and plain_text[pos] are
    alphanumeric characters).
    """
    n = len(plain_text)
    for pos, _ in tag_positions:
        if 0 < pos < n:
            if plain_text[pos - 1].isalnum() and plain_text[pos].isalnum():
                return False
    return True


def verify_and_align_paragraph(
    original_text: str,
    annotated_xml: str,
    min_similarity: float = 0.98,
) -> Optional[str]:
    """
    Verifies whether `annotated_xml`:
      1. Is well-formed XML with valid, non-empty `S`, `A`, `bystanders`, and `indirectspeech` attributes.
      2. Has no tags placed mid-word (e.g., splitting 'Hello' into 'Hel<turn>lo').
      3. Matches `original_text` exactly after stripping tags, OR differs ONLY in minor
         whitespace / quotation / dash characters (never adding, deleting, or replacing
         any alphanumeric characters via SequenceMatcher opcodes!).
    """
    if not annotated_xml or "<turn" not in annotated_xml:
        return None

    fixed_amp = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)", "&amp;", annotated_xml)
    try:
        p_check = ET.fromstring(f"<p>{fixed_amp}</p>")
        turns = p_check.findall(".//turn")
        if not turns:
            return None
        for t_elem in turns:
            validate_turn_attributes_strict(t_elem)
        annotated_xml = fixed_amp
    except (ET.ParseError, ValueError):
        return None

    tokens = ANY_TAG_RE.split(annotated_xml)
    tags = ANY_TAG_RE.findall(annotated_xml)

    tag_positions: List[Tuple[int, str]] = []
    reconstructed = ""
    for i, tok in enumerate(tokens):
        reconstructed += html.unescape(tok)
        if i < len(tags):
            tag_positions.append((len(reconstructed), tags[i]))

    # Case 1: Exact text match — still must pass the mid-word boundary check!
    if reconstructed == original_text:
        if not check_tag_boundaries_not_midword(original_text, tag_positions):
            return None
        return annotated_xml

    # Case 2: Minor non-alphanumeric differences — inspect SequenceMatcher opcodes strictly
    matcher = difflib.SequenceMatcher(None, reconstructed, original_text, autojunk=False)
    if matcher.ratio() < min_similarity:
        return None

    pos_map: Dict[int, int] = {0: 0, len(reconstructed): len(original_text)}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range((i2 - i1) + 1):
                pos_map[i1 + offset] = j1 + offset
        else:
            sub_a = reconstructed[i1:i2]
            sub_b = original_text[j1:j2]
            # Reject immediately if any letter or digit was added, deleted, or modified
            if any(ch.isalnum() for ch in sub_a) or any(ch.isalnum() for ch in sub_b):
                return None
            if sub_a and not ALLOWED_DIFF_CHARS_RE.match(sub_a):
                return None
            if sub_b and not ALLOWED_DIFF_CHARS_RE.match(sub_b):
                return None
            pos_map[i1] = j1
            pos_map[i2] = j2

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

    mapped_tag_positions: List[Tuple[int, str]] = []
    mapped_tags: Dict[int, List[str]] = {}
    for pos, tag in tag_positions:
        orig_pos = max(0, min(len(original_text), map_pos(pos)))
        mapped_tag_positions.append((orig_pos, tag))
        mapped_tags.setdefault(orig_pos, []).append(tag)

    # Verify projected tag boundaries are never mid-word in original_text
    if not check_tag_boundaries_not_midword(original_text, mapped_tag_positions):
        return None

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
        p_check2 = ET.fromstring(f"<p>{realigned}</p>")
        if "".join(p_check2.itertext()) != original_text:
            return None
        for t_elem in p_check2.findall(".//turn"):
            validate_turn_attributes_strict(t_elem)
        return realigned
    except (ET.ParseError, ValueError):
        return None


# =====================================================================
# 7. Gemini Batch Annotation with Sliding Context & Confidence Scoring
# =====================================================================

SYSTEM_INSTRUCTION = """You are a computational literary annotator specializing in the works of Jane Austen.
Your task is to annotate paragraphs from a Jane Austen novel by inserting inline <turn> and <interruption> XML tags according to the strict XML specification below, including a calibrated `confidence` score (0.00 to 1.00) on every element.

XML TAG & ATTRIBUTE SPECIFICATION:
- `<turn S="..." A="..." bystanders="..." [indirectspeech="IS|FID"] confidence="0.XX">...</turn>`: Wraps a unit of speech, letter, or thought.
  * `S` (Speaker): REQUIRED non-empty canonical name(s) of the speaker(s) or writer(s).
  * `A` (Addressee): REQUIRED non-empty canonical name(s) of the person(s) directly addressed or reading the letter.
  * `bystanders`: REQUIRED non-empty attribute listing canonical names of all other characters physically present in the scene/room who can hear the exchange, separated alphabetically by ` + `. If no other characters are present besides `S` and `A`, you MUST set `bystanders="no"` (NEVER leave `bystanders` empty or set to `"yes"`).
  * `indirectspeech` (Optional): Omit this attribute completely for direct speech. For non-direct speech, set ONLY to `"IS"` (Indirect Speech) or `"FID"` (Free Indirect Discourse).
  * `confidence`: A float formatted between `"0.00"` and `"1.00"` indicating your certainty about `S`, `A`, `bystanders`, speech type (`DS`/`IS`/`FID`), and exact span boundaries for this specific turn.
- `<interruption confidence="0.XX">...</interruption>`: Placed ONLY inside a `<turn>` element to wrap a narrative reporting clause that interrupts a character's speech mid-turn, with its own confidence score `"0.00"`..`"1.00"`.

CONFIDENCE CALIBRATION GUIDELINES:
- `0.95 - 1.00`: Unambiguous direct speech or explicit indirect speech where `S`, `A`, and `bystanders` are explicitly clear from the immediate text.
- `0.85 - 0.94`: Clear turn or indirect speech, but minor inference was required for the exact addressee subset or bystander list in a larger room.
- `0.65 - 0.84`: Ambiguous case (e.g., borderline Free Indirect Discourse vs. narrator commentary, uncertain room presence of a silent bystander, or unattributed turn in a multi-party conversation).
- `< 0.65`: Highly uncertain attribution or boundary.

STRICT ANNOTATION RULES:
1. UNMODIFIED TEXT & COMPLETE INDICES:
   - NEVER alter, add, or delete a single word, character, or punctuation mark of the original paragraph. Only insert `<turn ...>...</turn>` and `<interruption ...>...</interruption>` tags at valid word/punctuation boundaries. Escape bare `&` characters in the original text as `&amp;`.
   - You MUST return an entry in `paragraphs` for EVERY single `[Target p=idx]` in the input batch using its exact integer `paragraph_index`.
   - Names in `S`, `A`, and `bystanders` MUST ALWAYS use the normalized canonical name from the provided character list (e.g., 'Mrs. Elizabeth Bennet' instead of 'Lizzy' or 'Elizabeth').
2. DIRECT SPEECH & INTERRUPTIONS:
   - Wrap each spoken turn (including its enclosing quotation marks) in `<turn S="..." A="..." bystanders="..." confidence="0.XX">...</turn>`.
   - If a turn is interrupted mid-speech by a narrative reporting clause, wrap that clause in `<interruption confidence="0.XX">...</interruption>` INSIDE the `<turn>`:
     Example: `<turn S="Mrs. Bennet" A="Mr. Bennet" bystanders="no" confidence="0.99">"My dear Mr. Bennet," <interruption confidence="0.99">said his lady to him one day,</interruption> "have you heard that Netherfield Park is let at last?"</turn>`
   - If the narrative reporting clause appears completely BEFORE or AFTER the spoken words (outside the quotation marks), leave it OUTSIDE the `<turn>` tag:
     Example: `<turn S="Mrs. Bennet" A="Mr. Bennet" bystanders="no" confidence="0.99">"Do not you want to know who has taken it?"</turn> cried his wife impatiently.`
   - If the same speaker switches addressees (`A`) within the same paragraph (e.g., speaking first to Mr. Bennet and then turning to Lydia), close the first `<turn>` and open a new `<turn>` with the new `A` and updated `bystanders`.
3. INDIRECT SPEECH (`IS`) vs. FREE INDIRECT DISCOURSE (`FID`):
   - `indirectspeech="IS"` (Indirect Speech): Reported speech introduced by a reporting verb or conjunction (e.g., `Mr. Bennet replied that <turn S="Mr. Bennet" A="Mrs. Bennet" bystanders="no" indirectspeech="IS" confidence="0.98">he had not.</turn>`, or `Her aunt assured her <turn S="Mrs. Gardiner" A="Mrs. Elizabeth Bennet" bystanders="no" indirectspeech="IS" confidence="0.96">that she was</turn>`). Include short indirect clauses as well. If a reporting clause interrupts an indirect speech span, wrap the interrupting words in `<interruption confidence="0.XX">...</interruption>`.
   - `indirectspeech="FID"` (Free Indirect Discourse): Passages where the third-person narrative voice merges with a character's internal consciousness, thoughts, or speech stream without an explicit reporting clause (retaining third-person pronouns and past tense, but adopting the character's idiom, exclamations, rhetorical questions, or evaluative perspective, e.g., `<turn S="Mrs. Elizabeth Bennet" A="Mrs. Elizabeth Bennet" bystanders="no" indirectspeech="FID" confidence="0.88">What could be the meaning of it?--It was impossible to imagine; it was impossible not to long to know.</turn>`).
4. LETTERS & NOTES:
   - Every paragraph of a quoted letter or note (including salutations, body paragraphs, and sign-offs/signatures like `"Yours, &amp;c."` or `"FITZWILLIAM DARCY."`) MUST be wrapped in `<turn S="..." A="..." bystanders="..." confidence="0.XX">...</turn>`.
   - If a letter is read silently by the recipient, set `S` to the letter writer, `A` to the recipient reading it, and `bystanders="no"`.
   - If a letter is read aloud to others in a room, set `S` to the person reading it aloud and `A` / `bystanders` to the listeners present in the room.
5. SOLILOQUY / SPEAKING TO ONESELF:
   - When a character speaks, mutters, or exclaims to themselves (e.g., `"Insolent girl!" said Elizabeth to herself`, or `Elizabeth privately added, "And how much I shall have to conceal."`), annotate it as a `<turn>` with `S` and `A` both set to that character's canonical name and `bystanders="no"`.
6. MULTIPLE CHARACTERS:
   - If multiple characters speak in unison or multiple people are addressed simultaneously, separate their canonical names alphabetically with ` + ` (e.g., `A="Miss Catherine Bennet + Miss Lydia Bennet"`).
   - Ensure no character listed in `S` or `A` is repeated in `bystanders` for that same turn.
"""


def annotate_batch(
    client: GeminiAnnotatorClient,
    canonical_characters: List[str],
    context_paragraphs: List[Tuple[int, str]],
    target_paragraphs: List[Tuple[int, str]],
    current_scene_characters: List[str],
    book_title: str = "",
    max_retries: int = 4,
) -> Tuple[Dict[int, str], Dict[int, float], Set[int], List[str]]:
    """
    Annotates a batch of target paragraphs using Gemini and returns:
      - `annotated_map`: {paragraph_idx: annotated_xml}
      - `paragraph_confidences`: {paragraph_idx: unrounded_float}
      - `validated_empty`: set of paragraph_idx explicitly verified as having no speech
      - `latest_scene`: updated list of characters present at the end of the batch
    Enforces strict 1-to-1 index completeness, rejects inconsistent `has_speech`/`annotated_xml`
    combinations, and rejects missed quote paragraphs on ALL attempts (no final-attempt backdoor).
    """
    char_list_str = ", ".join(sorted(set(canonical_characters)))
    context_str = "\n".join([f"[Context p={idx}] {text}" for idx, text in context_paragraphs])
    target_str = "\n".join([f"[Target p={idx}] {text}" for idx, text in target_paragraphs])
    scene_str = ", ".join(current_scene_characters) if current_scene_characters else "None / To be inferred from the text"

    prompt = f"""CANONICAL CHARACTER LIST (use these exact names for S, A, and bystanders):
{char_list_str}

CHARACTERS PRESENT IN THE SCENE AT THE START OF THIS BATCH:
{scene_str}

PRECEDING CONTEXT (for reference only, to track who is in the room and who is speaking):
{context_str}

TARGET PARAGRAPHS TO ANNOTATE (return an entry in `paragraphs` for EVERY Target p=idx using the exact integer p=idx):
{target_str}
"""

    expected_indices = [idx for idx, _ in target_paragraphs]
    expected_set = set(expected_indices)
    orig_lookup = dict(target_paragraphs)

    last_error: Optional[Exception] = None
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
                raise ValueError("Empty response from model.")

            # Fallback if the model returned 0..N-1 relative indices instead of target_paragraphs indices
            returned_indices = [int(item.get("paragraph_index", -1)) for item in paragraphs_out]
            if (
                len(paragraphs_out) == len(target_paragraphs)
                and returned_indices == list(range(len(target_paragraphs)))
                and target_paragraphs[0][0] != 0
            ):
                for item, (real_idx, _) in zip(paragraphs_out, target_paragraphs):
                    item["paragraph_index"] = real_idx
                returned_indices = [int(item.get("paragraph_index", -1)) for item in paragraphs_out]

            # Strict 1-to-1 index completeness check (no missing, duplicate, or unexpected indices)
            if len(returned_indices) != len(expected_indices) or set(returned_indices) != expected_set:
                missing = sorted(expected_set - set(returned_indices))
                extra = sorted(set(returned_indices) - expected_set)
                raise ValueError(
                    f"Index mismatch in batch response: expected {len(expected_indices)} unique indices, "
                    f"got {len(returned_indices)} (missing={missing}, extra={extra})."
                )

            annotated_map: Dict[int, str] = {}
            paragraph_confidences: Dict[int, float] = {}
            validated_empty: Set[int] = set()
            latest_scene: List[str] = list(current_scene_characters)
            missed_quote_paragraphs: List[int] = []

            items_by_idx = {int(item["paragraph_index"]): item for item in paragraphs_out}
            for p_idx in expected_indices:
                item = items_by_idx[p_idx]
                if "scene_characters_present" in item and item["scene_characters_present"] is not None:
                    latest_scene = list(item["scene_characters_present"])

                has_speech = bool(item.get("has_speech", False))
                ann_xml = (item.get("annotated_xml") or "").strip()
                raw_p_conf, _ = format_confidence(item.get("paragraph_confidence", 0.90), default=0.90)
                paragraph_confidences[p_idx] = raw_p_conf

                orig_text = orig_lookup[p_idx]

                # Reject inconsistent combinations of has_speech and annotated_xml
                if has_speech and not ann_xml:
                    raise ValueError(
                        f"Paragraph p={p_idx} has has_speech=True but empty annotated_xml."
                    )
                if not has_speech and "<turn" in ann_xml:
                    raise ValueError(
                        f"Paragraph p={p_idx} has has_speech=False but annotated_xml contains <turn>."
                    )

                if has_speech:
                    aligned = verify_and_align_paragraph(orig_text, ann_xml)
                    if aligned is None:
                        raise ValueError(
                            f"XML alignment, boundary, or attribute validation failed for paragraph p={p_idx}."
                        )
                    annotated_map[p_idx] = aligned
                else:
                    if '"' in orig_text and not is_boilerplate_paragraph(orig_text, book_title):
                        missed_quote_paragraphs.append(p_idx)
                    else:
                        validated_empty.add(p_idx)

            # Unconditional rejection of missed quote paragraphs on EVERY attempt
            if missed_quote_paragraphs:
                raise ValueError(
                    f"Model missed direct speech/quotes in paragraph(s) {missed_quote_paragraphs}."
                )

            return annotated_map, paragraph_confidences, validated_empty, latest_scene

        except Exception as e:
            last_error = e
            wait = 2 ** (attempt + 1)
            print(f"  [Attempt {attempt + 1}/{max_retries} failed: {e}] Waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)

    raise RuntimeError(
        f"Batch failed after {max_retries} attempts (last error: {last_error}). "
        f"Progress has been saved in the checkpoint file; re-run the script to resume."
    )


# =====================================================================
# 8. Post-processing: Context-Aware Alias Resolution & Fail-Fast Export
# =====================================================================

def resolve_unknown_aliases_with_llm(
    client: GeminiAnnotatorClient,
    title: str,
    unknown_names_with_context: Dict[str, str],
    alias_map: Dict[str, str],
) -> None:
    """
    Prompts Gemini after the main pass to resolve any remaining non-canonical
    character references to their proper canonical names, providing the surrounding
    paragraph snippet for disambiguation.
    """
    if not unknown_names_with_context:
        return
    known_canonicals = sorted(set(alias_map.values()))
    items_for_prompt = [
        {"raw_name": name, "context_snippet": snippet[:280]}
        for name, snippet in sorted(unknown_names_with_context.items())
    ]
    prompt = f"""In our XML annotation of "{title}" by Jane Austen, the following raw character
references appeared (shown with their surrounding paragraph context):
{json.dumps(items_for_prompt, ensure_ascii=False, indent=2)}

Existing canonical character names:
{json.dumps(known_canonicals, ensure_ascii=False)}

Map each `raw_name` to the appropriate canonical name from the list above (or create a clean,
formal canonical name if it refers to a new minor character). Use the `context_snippet` to
disambiguate shared surnames or titles (e.g. 'Miss Bennet', 'Mr. Dashwood').
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
        print(f"[Warning] Could not resolve additional aliases: {e}", file=sys.stderr)


def serialize_xml_without_inner_whitespace(tree: ET.ElementTree, output_path: Path) -> None:
    """
    Serializes the XML tree so that each `<p>` is on its own indented line (`  <p>...</p>`),
    while NEVER inserting newlines or indentation inside `<p>` elements (preventing
    `ET.indent()` from corrupting paragraph text).
    """
    root = tree.getroot()
    root.text = "\n  "
    p_elems = list(root)
    for i, p_elem in enumerate(p_elems):
        p_elem.tail = "\n  " if i < len(p_elems) - 1 else "\n"
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def verify_xml_text_preservation(output_path: Path, original_paragraphs: List[str]) -> None:
    """
    Reads back the written XML file and verifies that the plain text of every `<p>`
    matches `original_paragraphs` 100% character-for-character.
    """
    parsed = ET.parse(output_path)
    p_elems = parsed.getroot().findall("p")
    if len(p_elems) != len(original_paragraphs):
        raise RuntimeError(
            f"Paragraph count mismatch in final XML: expected {len(original_paragraphs)}, got {len(p_elems)}."
        )
    for idx, (p_elem, orig_text) in enumerate(zip(p_elems, original_paragraphs)):
        reconstructed = "".join(p_elem.itertext())
        if reconstructed != orig_text:
            raise RuntimeError(
                f"Text preservation check failed at paragraph p={idx}:\n"
                f"  Expected: {orig_text[:120]!r}\n"
                f"  Got:      {reconstructed[:120]!r}"
            )


def build_final_xml(
    title: str,
    author: str,
    model_name: str,
    paragraphs: List[str],
    start_idx: int,
    end_idx: int,
    annotated_paragraphs: Dict[int, str],
    validated_empty_paragraphs: Set[int],
    paragraph_confidences: Dict[int, float],
    alias_map: Dict[str, str],
    client: Optional[GeminiAnnotatorClient] = None,
) -> Tuple[ET.ElementTree, Dict[str, int]]:
    """
    Builds the final XML tree with fail-fast verification:
      - Verifies every literary paragraph in `start_idx..end_idx-1` is either in
        `annotated_paragraphs` or `validated_empty_paragraphs`.
      - Raises an explicit RuntimeError if any annotated paragraph fails XML parsing
        or attribute validation (never silently dropping annotations).
      - Attaches `confidence="0.XX"` to `<p>`, `<turn>`, and `<interruption>` elements
        without `needs_review` attributes.
    """
    unverified = [
        idx
        for idx in range(start_idx, end_idx)
        if idx not in annotated_paragraphs and idx not in validated_empty_paragraphs
    ]
    if unverified:
        raise RuntimeError(
            f"Cannot export XML: {len(unverified)} literary paragraph(s) have neither annotations "
            f"nor validated_empty status (first 10: {unverified[:10]})."
        )

    root = ET.Element("book")
    root.set("title", title)
    root.set("author", author)
    root.set("model", model_name)

    unknown_names_with_context: Dict[str, str] = {}
    for idx, orig_p in enumerate(paragraphs):
        if idx in annotated_paragraphs:
            xml_p_str = f"<p>{annotated_paragraphs[idx]}</p>"
            try:
                p_elem = ET.fromstring(xml_p_str)
            except ET.ParseError as exc:
                raise RuntimeError(
                    f"Corrupted XML annotation in paragraph p={idx}: {exc}\nContent: {annotated_paragraphs[idx]!r}"
                ) from exc

            turns = p_elem.findall(".//turn")
            if not turns:
                raise RuntimeError(
                    f"Paragraph p={idx} is in annotated_paragraphs but contains no <turn> elements."
                )

            for turn in turns:
                validate_turn_attributes_strict(turn)
                for attr in ("S", "A", "bystanders"):
                    val = turn.get(attr, "")
                    for part in re.split(r"\s*\+\s*", val):
                        p_clean = part.strip()
                        if p_clean and p_clean.lower() not in {"no", "yes", "unknown", "?"}:
                            if p_clean.lower() not in alias_map and p_clean not in unknown_names_with_context:
                                unknown_names_with_context[p_clean] = orig_p
            root.append(p_elem)
        else:
            p_elem = ET.Element("p")
            p_elem.text = orig_p
            root.append(p_elem)

    if client and unknown_names_with_context:
        resolve_unknown_aliases_with_llm(client, title, unknown_names_with_context, alias_map)

    turn_counter = 1
    stats = {
        "total_turns": 0,
        "total_interruptions": 0,
    }

    for idx, p_elem in enumerate(root.findall("p")):
        if idx < start_idx or idx >= end_idx:
            # Front/back Gutenberg boilerplate outside the literary body
            default_p_conf = 1.00
        elif idx in annotated_paragraphs:
            default_p_conf = 0.90
        else:
            default_p_conf = 0.95

        base_p_conf = paragraph_confidences.get(idx, default_p_conf)
        min_elem_conf = base_p_conf

        for turn in p_elem.findall(".//turn"):
            raw_s = turn.get("S")
            raw_a = turn.get("A")
            raw_b = turn.get("bystanders")
            indirect_raw = turn.get("indirectspeech", None)
            raw_t_conf = turn.get("confidence", base_p_conf)

            norm_s = normalize_person_field(raw_s, alias_map, is_bystander=False)
            norm_a = normalize_person_field(raw_a, alias_map, is_bystander=False)

            active_participants = set(norm_s.split(" + ")) | set(norm_a.split(" + "))
            norm_b = normalize_person_field(
                raw_b, alias_map, exclude_names=active_participants, is_bystander=True
            )
            norm_indirect = normalize_indirect_speech_attr(indirect_raw, strict=True)
            t_conf_float, t_conf_str = format_confidence(raw_t_conf, default=base_p_conf)

            if "Unknown" in norm_s or "Unknown" in norm_a:
                t_conf_float = min(t_conf_float, 0.50)
                t_conf_str = f"{t_conf_float:.2f}"

            min_elem_conf = min(min_elem_conf, t_conf_float)

            turn.attrib.clear()
            turn.set("n", str(turn_counter))
            turn.set("S", norm_s)
            turn.set("A", norm_a)
            turn.set("bystanders", norm_b)
            if norm_indirect is not None:
                turn.set("indirectspeech", norm_indirect)
            turn.set("confidence", t_conf_str)

            for intr in turn.findall(".//interruption"):
                raw_i_conf = intr.get("confidence", t_conf_float)
                i_conf_float, i_conf_str = format_confidence(raw_i_conf, default=t_conf_float)
                min_elem_conf = min(min_elem_conf, i_conf_float)
                intr.attrib.clear()
                intr.set("confidence", i_conf_str)
                stats["total_interruptions"] += 1

            turn_counter += 1
            stats["total_turns"] += 1

        _, p_conf_str = format_confidence(min_elem_conf, default=base_p_conf)
        p_elem.set("confidence", p_conf_str)

    return ET.ElementTree(root), stats


# =====================================================================
# 9. Checkpoint Loader with Atomic Validation & Hash/Config Verification
# =====================================================================

def load_and_validate_checkpoint(
    ckpt_path: Path,
    expected_sha256: str,
    expected_batch_size: int,
    expected_model: str,
    start_idx: int,
) -> Tuple[
    Dict[int, str],
    Dict[int, float],
    Set[int],
    Dict[str, Dict[str, List[str]]],
    int,
    List[str],
    Optional[Dict[str, str]],
]:
    """
    Loads and validates the checkpoint into temporary variables before committing state.
    Raises RuntimeError if `input_sha256` mismatches the current input file.
    """
    raw_ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    if not isinstance(raw_ckpt, dict):
        raise ValueError("Checkpoint root is not a JSON object.")

    ckpt_sha = raw_ckpt.get("input_sha256")
    if ckpt_sha and ckpt_sha != expected_sha256:
        raise RuntimeError(
            f"Checkpoint input_sha256 mismatch in '{ckpt_path}'!\n"
            f"  Checkpoint SHA-256: {ckpt_sha}\n"
            f"  Current file SHA-256: {expected_sha256}\n"
            f"Refusing to load checkpoint from a modified input text. Use --fresh to start over."
        )

    ckpt_model = raw_ckpt.get("model")
    if ckpt_model and ckpt_model != expected_model:
        print(
            f"      [Notice] Checkpoint was created with model '{ckpt_model}', current model is '{expected_model}'.",
            file=sys.stderr,
        )

    ckpt_batch_size = raw_ckpt.get("batch_size")
    tmp_batch_states: Dict[str, Dict[str, List[str]]] = raw_ckpt.get("batch_scene_states", {})
    if ckpt_batch_size is not None and int(ckpt_batch_size) != expected_batch_size:
        print(
            f"      [Notice] Checkpoint batch_size ({ckpt_batch_size}) differs from current --batch-size "
            f"({expected_batch_size}); clearing batch_scene_states to recompute boundaries cleanly.",
            file=sys.stderr,
        )
        tmp_batch_states = {}

    tmp_annotated = {int(k): str(v) for k, v in raw_ckpt.get("annotated_paragraphs", {}).items()}
    tmp_confs = {int(k): float(v) for k, v in raw_ckpt.get("paragraph_confidences", {}).items()}
    tmp_empty = {int(x) for x in raw_ckpt.get("validated_empty_paragraphs", [])}
    tmp_last_idx = int(raw_ckpt.get("last_processed_idx", start_idx - 1))
    tmp_scene = [str(c) for c in raw_ckpt.get("current_scene_characters", [])]
    tmp_alias: Optional[Dict[str, str]] = None
    if raw_ckpt.get("alias_map") and isinstance(raw_ckpt["alias_map"], dict):
        tmp_alias = {str(k): str(v) for k, v in raw_ckpt["alias_map"].items()}

    if not tmp_empty and tmp_annotated and tmp_last_idx >= start_idx:
        print(
            "      [Notice] Migrating legacy checkpoint (inferring validated_empty for quote-free paragraphs).",
            file=sys.stderr,
        )

    return (
        tmp_annotated,
        tmp_confs,
        tmp_empty,
        tmp_batch_states,
        tmp_last_idx,
        tmp_scene,
        tmp_alias,
    )


# =====================================================================
# 10. CLI Entrypoint
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate a novel by Jane Austen (.txt) into a normalized XML file in output/."
    )
    parser.add_argument(
        "input_txt",
        type=Path,
        help="Path to the input .txt file of the book (e.g., pride_and_prejudice.txt)",
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        type=Path,
        default=Path("output"),
        help="Directory where *_checkpoint.json and *_annotated.xml are stored (default: output/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional explicit path for the output XML file (default: output/<input_stem>_annotated.xml)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gemini-4-argon",
        help="Gemini model name (default: gemini-4-argon)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of paragraphs per API call (default: 20)",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=8,
        help="Number of preceding paragraphs included as sliding context window (default: 8)",
    )
    parser.add_argument(
        "--upgrade-is-fid",
        action="store_true",
        help=(
            "Also re-process batches from an existing checkpoint that still use legacy "
            'indirectspeech="yes" so they get upgraded to IS/FID.'
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any existing checkpoint and re-annotate the entire book from scratch.",
    )
    args = parser.parse_args()

    if not args.input_txt.exists():
        print(f"Error: File not found: {args.input_txt}", file=sys.stderr)
        sys.exit(1)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_xml = args.output or (output_dir / f"{args.input_txt.stem}_annotated.xml")
    checkpoint_file = output_dir / f"{args.input_txt.stem}_checkpoint.json"
    legacy_checkpoint_file = args.input_txt.with_name(f"{args.input_txt.stem}_checkpoint.json")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "eu")

    if not api_key and not project_id:
        print(
            "Error: Neither GOOGLE_CLOUD_PROJECT (for Vertex AI / Cloud credits) nor GEMINI_API_KEY is set.\n"
            "  For Vertex AI (using Google Cloud credits):\n"
            "    export GOOGLE_CLOUD_PROJECT='your-project-id'\n"
            "    export GOOGLE_CLOUD_LOCATION='eu'\n"
            "  Or for Google AI Studio:\n"
            "    export GEMINI_API_KEY='your-api-key'",
            file=sys.stderr,
        )
        sys.exit(1)

    client = GeminiAnnotatorClient(
        api_key=api_key,
        project_id=project_id,
        location=location,
        model_name=args.model,
    )

    input_sha256 = compute_file_sha256(args.input_txt)
    mode_label = f"Vertex AI (project={project_id}, location={location})" if project_id else "Google AI Studio API"
    print(f"[1/4] Reading and parsing '{args.input_txt}'...")
    title, author, paragraphs, start_idx, end_idx = parse_gutenberg_text(args.input_txt)
    print(
        f"      Title: {title} | Author: {author}\n"
        f"      Backend: {mode_label} | Model: {args.model}\n"
        f"      Total paragraphs: {len(paragraphs)} (Literary body: {start_idx} to {end_idx - 1})\n"
        f"      Output directory: {output_dir.resolve()}"
    )

    annotated_paragraphs: Dict[int, str] = {}
    paragraph_confidences: Dict[int, float] = {}
    validated_empty_paragraphs: Set[int] = set()
    batch_scene_states: Dict[str, Dict[str, List[str]]] = {}
    last_processed_idx = start_idx - 1
    current_scene_characters: List[str] = []
    cached_alias_map: Optional[Dict[str, str]] = None

    if not args.fresh:
        ckpt_to_load = checkpoint_file if checkpoint_file.exists() else (
            legacy_checkpoint_file if legacy_checkpoint_file.exists() else None
        )
        if ckpt_to_load is not None:
            try:
                (
                    annotated_paragraphs,
                    paragraph_confidences,
                    validated_empty_paragraphs,
                    batch_scene_states,
                    last_processed_idx,
                    current_scene_characters,
                    cached_alias_map,
                ) = load_and_validate_checkpoint(
                    ckpt_path=ckpt_to_load,
                    expected_sha256=input_sha256,
                    expected_batch_size=args.batch_size,
                    expected_model=args.model,
                    start_idx=start_idx,
                )
                if not validated_empty_paragraphs and annotated_paragraphs and last_processed_idx >= start_idx:
                    for i in range(start_idx, min(end_idx, last_processed_idx + 1)):
                        if i not in annotated_paragraphs and (
                            '"' not in paragraphs[i] or is_boilerplate_paragraph(paragraphs[i], title)
                        ):
                            validated_empty_paragraphs.add(i)
                print(
                    f"      Loaded checkpoint from '{ckpt_to_load}' "
                    f"({len(annotated_paragraphs)} annotated, {len(validated_empty_paragraphs)} validated empty, "
                    f"last_processed_idx={last_processed_idx})."
                )
            except RuntimeError as hash_err:
                print(f"Error: {hash_err}", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"      [Warning] Could not read checkpoint ({e}); starting fresh.", file=sys.stderr)

    print(f"[2/4] Building canonical character registry with {args.model}...")
    literary_paragraphs = paragraphs[start_idx:end_idx]
    if cached_alias_map:
        alias_map = cached_alias_map
        print(f"      Restored {len(set(alias_map.values()))} canonical characters from checkpoint.")
    else:
        alias_map = build_character_registry(client, title, author, literary_paragraphs)
        print(f"      Registered {len(set(alias_map.values()))} canonical characters.")
    canonical_characters = sorted(set(alias_map.values()))

    print(f"[3/4] Annotating paragraphs in batches of {args.batch_size}...")
    running_scene: List[str] = []

    for batch_start in range(start_idx, end_idx, args.batch_size):
        batch_end = min(end_idx, batch_start + args.batch_size)
        batch_indices = list(range(batch_start, batch_end))
        batch_key = f"{batch_start}_{batch_end - 1}"

        already_covered = (batch_end - 1) <= last_processed_idx
        unannotated_quotes = [
            i for i in batch_indices
            if i not in annotated_paragraphs
            and '"' in paragraphs[i]
            and not is_boilerplate_paragraph(paragraphs[i], title)
        ]
        all_explicitly_validated = all(
            (i in annotated_paragraphs or i in validated_empty_paragraphs) for i in batch_indices
        )
        has_legacy_indirect = any(
            'indirectspeech="yes"' in annotated_paragraphs.get(i, "")
            for i in batch_indices
        )

        # Check scene-state continuity between previous batch's actual end state (`running_scene`)
        # and this batch's saved start state in `batch_scene_states`
        saved_state = batch_scene_states.get(batch_key)
        scene_state_mismatch = False
        if saved_state is not None and "start" in saved_state and batch_start > start_idx:
            if sorted(set(saved_state["start"])) != sorted(set(running_scene)):
                scene_state_mismatch = True

        needs_reprocess = False
        reason = "new batch"
        if not already_covered:
            needs_reprocess = True
            reason = "unprocessed batch"
        elif not all_explicitly_validated:
            needs_reprocess = True
            reason = "incomplete validation state in batch"
        elif unannotated_quotes:
            needs_reprocess = True
            reason = f"repairing {len(unannotated_quotes)} missed quote paragraph(s) {unannotated_quotes}"
        elif args.upgrade_is_fid and has_legacy_indirect:
            needs_reprocess = True
            reason = "upgrading indirectspeech='yes' to IS/FID"
        elif scene_state_mismatch:
            needs_reprocess = True
            reason = "cascading scene-state update from preceding batch"

        if not needs_reprocess:
            if saved_state and "end" in saved_state:
                running_scene = list(saved_state["end"])
            continue

        scene_at_batch_start = list(running_scene) if batch_start > start_idx else []

        ctx_start = max(start_idx, batch_start - args.context_size)
        context_batch = [(i, paragraphs[i]) for i in range(ctx_start, batch_start)]
        target_batch = [(i, paragraphs[i]) for i in batch_indices]

        print(f"      Processing paragraphs {batch_start}..{batch_end - 1} / {end_idx - 1} ({reason})...")
        batch_result, batch_confs, batch_empty, running_scene = annotate_batch(
            client=client,
            canonical_characters=canonical_characters,
            context_paragraphs=context_batch,
            target_paragraphs=target_batch,
            current_scene_characters=scene_at_batch_start,
            book_title=title,
        )

        for i in batch_indices:
            annotated_paragraphs.pop(i, None)
            validated_empty_paragraphs.discard(i)
        annotated_paragraphs.update(batch_result)
        paragraph_confidences.update(batch_confs)
        validated_empty_paragraphs.update(batch_empty)
        batch_scene_states[batch_key] = {
            "start": scene_at_batch_start,
            "end": list(running_scene),
        }
        last_processed_idx = max(last_processed_idx, batch_end - 1)
        current_scene_characters = list(running_scene)

        checkpoint_data = {
            "schema_version": SCHEMA_VERSION,
            "input_sha256": input_sha256,
            "model": args.model,
            "batch_size": args.batch_size,
            "last_processed_idx": last_processed_idx,
            "current_scene_characters": current_scene_characters,
            "alias_map": alias_map,
            "batch_scene_states": batch_scene_states,
            "annotated_paragraphs": {str(k): v for k, v in sorted(annotated_paragraphs.items())},
            "paragraph_confidences": {str(k): round(v, 4) for k, v in sorted(paragraph_confidences.items())},
            "validated_empty_paragraphs": sorted(validated_empty_paragraphs),
        }
        atomic_write_json(checkpoint_file, checkpoint_data)

    print(f"[4/4] Normalizing character names, attaching confidence scores, and writing final XML to '{output_xml}'...")
    tree, stats = build_final_xml(
        title=title,
        author=author,
        model_name=args.model,
        paragraphs=paragraphs,
        start_idx=start_idx,
        end_idx=end_idx,
        annotated_paragraphs=annotated_paragraphs,
        validated_empty_paragraphs=validated_empty_paragraphs,
        paragraph_confidences=paragraph_confidences,
        alias_map=alias_map,
        client=client,
    )
    serialize_xml_without_inner_whitespace(tree, output_xml)
    verify_xml_text_preservation(output_xml, paragraphs)

    print(
        f"Done! Annotated XML saved to: {output_xml}\n"
        f"      Total turns annotated:         {stats['total_turns']}\n"
        f"      Total interruptions annotated: {stats['total_interruptions']}\n"
        f"      Text preservation verified:    100% exact match across all {len(paragraphs)} paragraphs."
    )


if __name__ == "__main__":
    main()

import argparse
import glob
import html
import json
import math
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

TRAINING_SYSTEM_PROMPT = (
    "You are an expert NLP system specialized in 19th-century English literature. "
    "Your task is to annotate a given raw paragraph with structured XML dialogue tags. "
    "Precisely identify the Speaker (S), Addressee (A), Bystanders (yes/no), and Indirect Speech (yes). "
    "Wrap any dialogue in <turn> tags, and wrap narrative interruptions inside the dialogue in <interruption> tags. "
    "If the paragraph contains no dialogue, output it exactly as is. "
    "Rely strictly on the textual structure and punctuation provided. Do not use prior knowledge of the novel."
)

TURN_RE = re.compile(r"<turn\s+([^>]*)>(.*?)</turn>", re.IGNORECASE | re.DOTALL)
INTERRUPTION_RE = re.compile(r"<interruption>(.*?)</interruption>", re.IGNORECASE | re.DOTALL)
ATTRIBUTE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([\"'])(.*?)\2", re.DOTALL)
ANNOTATION_TAG_RE = re.compile(r"</?(?:turn|interruption)\b[^>]*>", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate all input text files with a fine-tuned Llama 70B LoRA adapter."
    )
    parser.add_argument("--base_model_id", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--input_dir", default="input")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--file_pattern", default="*.txt")
    parser.add_argument("--context_paragraphs", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_paragraphs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def read_paragraphs(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    return [
        re.sub(r"[ \t]*\n[ \t]*", " ", part.strip())
        for part in re.split(r"\n\s*\n", text)
        if part.strip()
    ]


def extract_metadata(items, stem):
    title = None
    author = None
    for paragraph in items[:100]:
        title_match = re.match(r"^Title:\s*(.+)$", paragraph, re.IGNORECASE)
        author_match = re.match(r"^Author:\s*(.+)$", paragraph, re.IGNORECASE)
        if title_match and title is None:
            title = title_match.group(1).strip()
        if author_match and author is None:
            author = author_match.group(1).strip()
    return title or stem.replace("_", " ").title(), author or "Jane Austen"


def is_structural_heading(paragraph):
    text = paragraph.strip()
    patterns = (
        r"CHAPTER\s+[IVXLCDM0-9]+\.?",
        r"(?:VOLUME|VOL\.)\s+[IVXLCDM0-9]+\.?",
        r"BOOK\s+[IVXLCDM0-9]+\.?",
    )
    return any(re.fullmatch(pattern, text, re.IGNORECASE) for pattern in patterns)


def gutenberg_body_bounds(items):
    after_start = 0
    body_end = len(items)
    for index, paragraph in enumerate(items):
        marker = paragraph.strip().upper()
        if marker.startswith("*** START OF THE PROJECT GUTENBERG EBOOK") or marker.startswith(
            "*** START OF THIS PROJECT GUTENBERG EBOOK"
        ):
            after_start = index + 1
            break
    for index in range(after_start, len(items)):
        marker = items[index].strip().upper()
        if marker.startswith("*** END OF THE PROJECT GUTENBERG EBOOK") or marker.startswith(
            "*** END OF THIS PROJECT GUTENBERG EBOOK"
        ):
            body_end = index
            break
    body_start = after_start
    for index in range(after_start, body_end):
        if is_structural_heading(items[index]):
            body_start = index
            break
    return body_start, body_end


def build_device_map(model_id):
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("No CUDA GPU is available.")
    config = AutoConfig.from_pretrained(model_id)
    layer_count = getattr(config, "num_hidden_layers", None)
    if gpu_count == 1 or layer_count is None:
        return "auto"
    layers_per_gpu = math.ceil(layer_count / gpu_count)
    mapping = {"model.embed_tokens": 0, "model.norm": gpu_count - 1, "lm_head": gpu_count - 1}
    for layer_index in range(layer_count):
        mapping[f"model.layers.{layer_index}"] = min(layer_index // layers_per_gpu, gpu_count - 1)
    return mapping


def load_model(model_id, adapter_dir):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("The tokenizer does not define an EOS token.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"Loading on {torch.cuda.device_count()} GPU(s).", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=build_device_map(model_id),
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def build_stop_ids(tokenizer):
    result = []
    for token_id in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")):
        if isinstance(token_id, int) and token_id >= 0 and token_id not in result:
            result.append(token_id)
    return result


def generated_token_character_spans(tokenizer, generated_ids):
    spans = []
    previous_length = 0
    for end in range(1, len(generated_ids) + 1):
        decoded = tokenizer.decode(
            generated_ids[:end],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        spans.append((previous_length, len(decoded)))
        previous_length = len(decoded)
    return spans


def confidence_for_char_span(probabilities, token_spans, char_start, char_end):
    values = [
        probability
        for probability, (token_start, token_end) in zip(probabilities, token_spans)
        if token_end > char_start and token_start < char_end
    ]
    return min(values) if values else None


def find_attribute_value_span(opening_tag, attribute_name):
    pattern = re.compile(
        rf"\b{re.escape(attribute_name)}\s*=\s*([\"'])(.*?)\1",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(opening_tag)
    return match.span(2) if match else None


def parse_generated_turns(generated_text, probabilities, token_spans):
    turns = []
    for match in TURN_RE.finditer(generated_text):
        whole = match.group(0)
        opening_end_relative = whole.find(">") + 1
        opening_tag = whole[:opening_end_relative]
        attributes = {
            name: value.strip()
            for name, _, value in ATTRIBUTE_RE.findall(match.group(1))
        }
        opening_start = match.start()
        opening_end = match.start() + opening_end_relative
        detection_confidence = confidence_for_char_span(
            probabilities, token_spans, opening_start, opening_end
        )
        component_confidences = {}
        for name in ("S", "A", "bystanders"):
            relative_span = find_attribute_value_span(opening_tag, name)
            component_confidences[name] = (
                confidence_for_char_span(
                    probabilities,
                    token_spans,
                    opening_start + relative_span[0],
                    opening_start + relative_span[1],
                )
                if relative_span
                else None
            )
        indirect_attribute_name = None
        indirect_span = None
        indirect_value_raw = None

        for candidate_name in (
            "indirectspeech",
            "indirect_speech",
            "indirect",
        ):
            candidate_span = find_attribute_value_span(
                opening_tag,
                candidate_name,
            )

            if candidate_span is not None:
                indirect_attribute_name = candidate_name
                indirect_span = candidate_span
                indirect_value_raw = attributes.get(candidate_name)
                break

        if indirect_span is not None:
            indirect_value = (
                indirect_value_raw.strip().lower()
                if indirect_value_raw is not None
                else "yes"
            )

            if indirect_value in {"true", "1"}:
                indirect_value = "yes"
            elif indirect_value in {"false", "0"}:
                indirect_value = "no"
            elif indirect_value not in {"yes", "no"}:
                indirect_value = "unknown"

            indirect_confidence = confidence_for_char_span(
                probabilities,
                token_spans,
                opening_start + indirect_span[0],
                opening_start + indirect_span[1],
            )

            indirect_confidence_basis = (
                f"explicit_attribute:{indirect_attribute_name}"
            )
        else:
            indirect_value = "no"

            closing_bracket_start = opening_end - 1

            indirect_confidence = confidence_for_char_span(
                probabilities,
                token_spans,
                closing_bracket_start,
                opening_end,
            )

            indirect_confidence_basis = (
                "opening_tag_closed_without_"
                "indirectspeech_attribute"
            )
        available = [
            detection_confidence,
            component_confidences["S"],
            component_confidences["A"],
            component_confidences["bystanders"],
            indirect_confidence,
        ]
        overall_confidence = min(value for value in available if value is not None) if any(
            value is not None for value in available
        ) else None
        interruptions = []
        content_start = match.start(2)
        for interruption_match in INTERRUPTION_RE.finditer(match.group(2)):
            interruptions.append(
                {
                    "text": interruption_match.group(1),
                    "generated_start": content_start + interruption_match.start(1),
                    "generated_end": content_start + interruption_match.end(1),
                    "confidence": confidence_for_char_span(
                        probabilities,
                        token_spans,
                        content_start + interruption_match.start(),
                        content_start + interruption_match.end(),
                    ),
                }
            )
        turns.append(
            {
                "S": attributes.get("S", "Unknown"),
                "A": attributes.get("A", "Unknown"),
                "bystanders": attributes.get("bystanders", "Unknown").lower(),
                "indirectspeech": indirect_value,
                "content": match.group(2),
                "interruptions": interruptions,
                "confidence": overall_confidence,
                "turn_detection_confidence": detection_confidence,
                "S_confidence": component_confidences["S"],
                "A_confidence": component_confidences["A"],
                "bystanders_confidence": component_confidences["bystanders"],
                "indirectspeech_confidence": indirect_confidence,
                "indirectspeech_confidence_basis": indirect_confidence_basis,
            }
        )
    return turns


def generate_annotation(model, tokenizer, target, context, maximum, stop_token_ids):
    context_text = "\n\n".join(context) if context else "No preceding context (Start of book)."
    user_prompt = (
        f"Context history:\n{context_text}\n\n"
        f"Target paragraph to parse:\n{target}"
    )
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=maximum,
            do_sample=False,
            use_cache=True,
            eos_token_id=stop_token_ids,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    input_length = inputs["input_ids"].shape[1]
    generated_tensor = output.sequences[0, input_length:]
    generated_ids = generated_tensor.tolist()
    generated_text = tokenizer.decode(
        generated_tensor,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    probabilities = []
    for index, token_id in enumerate(generated_ids):
        log_probability = torch.log_softmax(output.scores[index][0].float(), dim=-1)[token_id]
        probabilities.append(float(log_probability.exp().item()))
    token_spans = generated_token_character_spans(tokenizer, generated_ids)
    turns = parse_generated_turns(generated_text, probabilities, token_spans)
    malformed = "<turn" in generated_text.lower() and not turns
    return generated_text, turns, malformed


def normalize_with_boundaries(text):
    normalized = []
    boundaries = [0]
    in_whitespace = False
    for index, character in enumerate(text):
        if character.isspace():
            if normalized and not in_whitespace:
                normalized.append(" ")
                boundaries.append(index + 1)
            in_whitespace = True
        else:
            normalized.append(character)
            boundaries.append(index + 1)
            in_whitespace = False
    if normalized and normalized[-1] == " ":
        normalized.pop()
        boundaries.pop()
    return "".join(normalized), boundaries


def strip_annotation_tags(text):
    return html.unescape(ANNOTATION_TAG_RE.sub("", text))


def map_interruptions_to_source(turn_source, interruptions):
    normalized_source, boundaries = normalize_with_boundaries(turn_source)
    cursor = 0
    mapped = []
    for interruption in interruptions:
        item = dict(interruption)
        normalized_text, _ = normalize_with_boundaries(html.unescape(interruption["text"]))
        position = normalized_source.find(normalized_text, cursor) if normalized_text else -1
        if position < 0:
            item["mapping_status"] = "not_found_in_source_turn"
        else:
            end = position + len(normalized_text)
            item.update(
                mapping_status="exact_normalized_match",
                source_start=boundaries[position],
                source_end=boundaries[end],
            )
            cursor = end
        mapped.append(item)
    return mapped


def map_turns_to_source(source, turns):
    normalized_source, boundaries = normalize_with_boundaries(source)
    cursor = 0
    mapped = []
    for turn in turns:
        item = dict(turn)
        candidate = strip_annotation_tags(turn["content"])
        normalized_candidate, _ = normalize_with_boundaries(candidate)
        position = normalized_source.find(normalized_candidate, cursor) if normalized_candidate else -1
        if position < 0:
            item["mapping_status"] = "not_found_in_source"
            item["candidate_text"] = candidate
            item["mapped_interruptions"] = []
        else:
            end = position + len(normalized_candidate)
            source_start = boundaries[position]
            source_end = boundaries[end]
            source_turn = source[source_start:source_end]
            item.update(
                mapping_status="exact_normalized_match",
                source_start=source_start,
                source_end=source_end,
                candidate_text=candidate,
                mapped_interruptions=map_interruptions_to_source(
                    source_turn, turn["interruptions"]
                ),
            )
            cursor = end
        mapped.append(item)
    return mapped


def confidence_text(value):
    return "unavailable" if value is None else f"{max(0.0, min(1.0, value)):.6f}"


def requires_entity_review(value):
    normalized = value.strip().lower()
    blocked = {
        "unknown", "narrator", "reader", "he", "she", "him", "her", "himself",
        "herself", "husband", "wife", "the lady", "lady", "previous speaker",
    }
    return normalized in blocked


def append_turn_mixed_content(turn_element, source_turn, interruptions):
    valid = [
        interruption
        for interruption in interruptions
        if interruption.get("mapping_status") == "exact_normalized_match"
    ]
    cursor = 0
    previous = None
    for interruption in valid:
        preceding = source_turn[cursor:interruption["source_start"]]
        if previous is None:
            turn_element.text = preceding
        else:
            previous.tail = preceding
        child = ET.SubElement(
            turn_element,
            "interruption",
            {"confidence": confidence_text(interruption.get("confidence"))},
        )
        child.text = source_turn[interruption["source_start"]:interruption["source_end"]]
        previous = child
        cursor = interruption["source_end"]
    trailing = source_turn[cursor:]
    if previous is None:
        turn_element.text = trailing
    else:
        previous.tail = trailing
    return len(valid) == len(interruptions)


def add_paragraph(book, source, turns, next_number):
    paragraph_element = ET.SubElement(book, "p")
    mapped = map_turns_to_source(source, turns)
    valid = [turn for turn in mapped if turn["mapping_status"] == "exact_normalized_match"]
    if not valid:
        paragraph_element.text = source
        if turns:
            paragraph_element.set("annotation_status", "mapping_failed")
        return next_number, mapped
    cursor = 0
    previous = None
    for turn in valid:
        preceding = source[cursor:turn["source_start"]]
        if previous is None:
            paragraph_element.text = preceding
        else:
            previous.tail = preceding
        speaker = turn["S"]
        addressee = turn["A"]
        overall_confidence = turn["confidence"]
        attributes = {
            "number": str(next_number),
            "confidence": confidence_text(turn["confidence"]),
            "turn_detection_confidence": confidence_text(turn["turn_detection_confidence"]),
            "S": speaker,
            "S_confidence": confidence_text(turn["S_confidence"]),
            "A": addressee,
            "A_confidence": confidence_text(turn["A_confidence"]),
            "bystanders": turn["bystanders"],
            "bystanders_confidence": confidence_text(turn["bystanders_confidence"]),
            "indirectspeech": turn["indirectspeech"],
            "indirectspeech_confidence": confidence_text(turn["indirectspeech_confidence"]),
            "indirectspeech_confidence_basis": turn["indirectspeech_confidence_basis"],
            "review_recommended": str(
                requires_entity_review(speaker) or requires_entity_review(addressee) or overall_confidence is None or overall_confidence < 0.50
            ).lower(),
        }
        turn_element = ET.SubElement(paragraph_element, "turn", attributes)
        source_turn = source[turn["source_start"]:turn["source_end"]]
        interruptions_complete = append_turn_mixed_content(
            turn_element, source_turn, turn["mapped_interruptions"]
        )
        if not interruptions_complete:
            turn_element.set("interruption_mapping_status", "partially_failed")
        previous = turn_element
        cursor = turn["source_end"]
        next_number += 1
    previous.tail = source[cursor:]
    if len(valid) != len(turns):
        paragraph_element.set("annotation_status", "partially_mapped")
    return next_number, mapped


def indent_xml(element, level=0):
    indentation = "\n" + "  " * level
    if len(element):
        if not element.text or not element.text.strip():
            element.text = indentation + "  "
        for child in element:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indentation
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indentation


def save_xml(book, output_path):
    indent_xml(book)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=output_path.parent, suffix=".tmp") as handle:
        temporary_path = Path(handle.name)
    try:
        ET.ElementTree(book).write(
            temporary_path,
            encoding="utf-8",
            xml_declaration=True,
            short_empty_elements=False,
        )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def open_or_create_book(output_path, title, author, overwrite):
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        root = ET.parse(output_path).getroot()
        processed = int(root.get("processed_paragraphs", "0"))
        numbers = [
            int(turn.get("number"))
            for turn in root.iter("turn")
            if (turn.get("number") or "").isdigit()
        ]
        return root, processed, max(numbers, default=0) + 1
    root = ET.Element(
        "book",
        title=title,
        author=author,
        model="Llama-3.3-70B-Instruct-LoRA",
        confidence_method="minimum_generated_token_probability",
        prompt_regime="identical_to_fine_tuning",
        annotation_scope="gutenberg_literary_body_only",
        processed_paragraphs="0",
    )
    return root, 0, 1


def process_book(model, tokenizer, source_path, args, stop_token_ids):
    source_path = Path(source_path)
    all_items = read_paragraphs(source_path)
    body_start, full_body_end = gutenberg_body_bounds(all_items)
    items = all_items[: args.max_paragraphs] if args.max_paragraphs is not None else all_items
    body_end = min(full_body_end, len(items))
    title, author = extract_metadata(all_items, source_path.stem)
    output_path = Path(args.output_dir) / f"{source_path.stem}_annotated_llama70B.xml"
    audit_path = Path(args.output_dir) / f"{source_path.stem}_annotated_llama70B_audit.jsonl"
    if args.overwrite or not output_path.exists():
        audit_path.unlink(missing_ok=True)
    book, start_index, next_number = open_or_create_book(
        output_path, title, author, args.overwrite
    )
    if start_index >= len(items):
        print(f"Skipping completed file: {output_path}", flush=True)
        return
    print(
        f"{source_path.name}: {len(items)} paragraphs; literary body "
        f"{body_start + 1}..{body_end}; starting at {start_index + 1}.",
        flush=True,
    )
    started = time.perf_counter()
    inferred_count = 0
    inferred_seconds = 0.0
    for index in range(start_index, len(items)):
        in_body = body_start <= index < body_end
        should_annotate = in_body and not is_structural_heading(items[index])
        malformed = False
        if should_annotate:
            context_start = max(body_start, index - args.context_paragraphs)
            inference_started = time.perf_counter()
            generated_text, turns, malformed = generate_annotation(
                model,
                tokenizer,
                items[index],
                items[context_start:index],
                args.max_new_tokens,
                stop_token_ids,
            )
            inferred_seconds += time.perf_counter() - inference_started
            inferred_count += 1
            processing_status = "malformed_model_output" if malformed else "model_inference"
            filtered = []
            for turn in turns:
                if turn["S"].strip().lower() == "narrator" and turn["A"].strip().lower() == "reader":
                    turn["rejection_reason"] = "narrator_reader_is_not_a_dialogue_turn"
                else:
                    filtered.append(turn)
            rejected_turns = [turn for turn in turns if "rejection_reason" in turn]
            turns = filtered
        else:
            generated_text = items[index]
            turns = []
            rejected_turns = []
            processing_status = (
                "outside_gutenberg_body" if not in_body else "structural_heading_skipped"
            )
        next_number, mapped_turns = add_paragraph(book, items[index], turns, next_number)
        book.set("processed_paragraphs", str(index + 1))
        with audit_path.open("a", encoding="utf-8") as audit_file:
            audit_file.write(
                json.dumps(
                    {
                        "paragraph_index": index,
                        "processing_status": processing_status,
                        "source_text": items[index],
                        "model_output": generated_text,
                        "turns": mapped_turns,
                        "rejected_turns": rejected_turns,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        if args.verbose and index < start_index + 3:
            print(
                f"Paragraph {index + 1}: status={processing_status}; "
                f"accepted={len(turns)}; rejected={len(rejected_turns)}; output={generated_text}",
                flush=True,
            )
        completed = index + 1
        if completed % args.checkpoint_every == 0 or completed == len(items):
            save_xml(book, output_path)
            remaining_inference = sum(
                body_start <= future < body_end and not is_structural_heading(items[future])
                for future in range(completed, len(items))
            )
            inference_rate = inferred_count / inferred_seconds if inferred_seconds > 0 else 0.0
            eta_minutes = (
                remaining_inference / inference_rate / 60 if inference_rate > 0 else 0.0
            )
            print(
                f"{source_path.name}: {completed}/{len(items)}; turns={next_number - 1}; "
                f"inferred={inferred_count}; ETA={eta_minutes:.1f} min.",
                flush=True,
            )
    print(
        f"Completed: {output_path} in {(time.perf_counter() - started) / 60:.1f} minutes.",
        flush=True,
    )


def main():
    args = parse_args()
    files = sorted(glob.glob(str(Path(args.input_dir) / args.file_pattern)))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError("No input text files were found.")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.base_model_id, args.adapter_dir)
    stop_token_ids = build_stop_ids(tokenizer)
    for source_path in files:
        process_book(model, tokenizer, source_path, args, stop_token_ids)


if __name__ == "__main__":
    main()


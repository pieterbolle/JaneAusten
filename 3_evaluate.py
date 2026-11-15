# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Pieter Bolle
import os
import re
import math
import torch
import argparse
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
from sklearn.metrics import classification_report

def extract_all_turns_metadata(xml_string):
    """
    Find all turns in a paragraph and return their metadata as a list of tuples.
    """
    if not isinstance(xml_string, str):
        return []
        
    turn_matches = re.findall(r'<turn\s+(.*?)>', xml_string)
    turns_metadata = []
    
    for meta in turn_matches:
        s_match = re.search(r'S=["\']([^"\']*)["\']', meta)
        a_match = re.search(r'A=["\']([^"\']*)["\']', meta)
        b_match = re.search(r'bystanders=["\']([^"\']*)["\']', meta)
        
        speaker = s_match.group(1) if s_match else "Unknown"
        addressee = a_match.group(1) if a_match else "Unknown"
        bystanders = b_match.group(1).lower() if b_match else "Unknown"
        turns_metadata.append((speaker, addressee, bystanders))
        
    return turns_metadata

def main():
    parser = argparse.ArgumentParser(description="Universal paragraph-level evaluation engine")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--data_file", type=str, default="intermediate_output/val.jsonl")
    parser.add_argument("--output_file", type=str, default="evaluation_results_val.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Show detailed turn-by-turn logging")
    args = parser.parse_args()

    print("==================================================")
    print(" UNIVERSAL EVALUATION ENGINE INITIALIZATION")
    print("==================================================")
    print(f"--> Base model:      {args.model_id}")
    print(f"--> LoRA adapters:   {args.adapter_dir}")
    print(f"--> Evaluation data: {args.data_file}")
    print(f"--> Results file:    {args.output_file}")
    print(f"--> Max new tokens:  {args.max_new_tokens}")
    print(f"--> Max samples:     {args.max_samples}")
    print(f"--> Verbose mode:    {args.verbose}")
    print("==================================================")
    
    dataset = load_dataset("json", data_files={"validation": args.data_file})
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    num_gpus = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(args.model_id)
    num_layers = getattr(config, "num_hidden_layers", None)

    if num_gpus > 1 and num_layers is not None:
        print(f"--> Multi-GPU setup detected ({num_gpus} GPUs). Building a static device_map for {num_layers} layers...")
        layers_per_gpu = math.ceil(num_layers / num_gpus)
        custom_device_map = {
            "model.embed_tokens": 0,
            "model.norm": num_gpus - 1,
            "lm_head": num_gpus - 1
        }
        for layer_idx in range(num_layers):
            gpu_idx = min(layer_idx // layers_per_gpu, num_gpus - 1)
            custom_device_map[f"model.layers.{layer_idx}"] = gpu_idx
        device_map_to_use = custom_device_map
    else:
        print("--> Single GPU or unknown architecture detected. Using device_map='auto'.")
        device_map_to_use = "auto"

    base_model_eval = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map_to_use,
        attn_implementation="flash_attention_2"
    )
    
    eval_model = PeftModel.from_pretrained(
        base_model_eval,
        args.adapter_dir,
        is_trainable=False,
    )
    eval_model.config.use_cache = True
    eval_model.eval()

    stop_tokens = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int):
        stop_tokens.append(eot_id)

    y_true_speaker, y_pred_speaker = [], []
    y_true_addressee, y_pred_addressee = [], []
    y_true_bystanders, y_pred_bystanders = [], []

    paragraph_exact_matches = 0
    turn_count_exact_matches = 0
    total_missing_turns = 0
    total_extra_turns = 0
    total_true_turns = 0
    total_pred_turns = 0

    eval_set = dataset["validation"]
    if args.max_samples is not None:
        eval_set = eval_set.select(range(min(args.max_samples, len(eval_set))))

    output_path = os.path.abspath(args.output_file)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    print("\n--- Starting generation and scoring on the paragraph dataset ---")

    for idx, sample in enumerate(eval_set):
        messages = sample["messages"]
        input_messages = messages[:-1]
        ground_truth_xml = messages[-1]["content"]
    
        prompt = tokenizer.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_device = eval_model.get_input_embeddings().weight.device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

        with torch.inference_mode():
            generated_sequences = eval_model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_tokens,
                use_cache=True,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generated_sequences[0, input_length:]
        generated_xml_raw = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        # Extract all turns from the paragraph.
        true_turns = extract_all_turns_metadata(ground_truth_xml)
        pred_turns = extract_all_turns_metadata(generated_xml_raw)

        total_true_turns += len(true_turns)
        total_pred_turns += len(pred_turns)
        total_missing_turns += max(0, len(true_turns) - len(pred_turns))
        total_extra_turns += max(0, len(pred_turns) - len(true_turns))
        if len(true_turns) == len(pred_turns):
            turn_count_exact_matches += 1
        if true_turns == pred_turns:
            paragraph_exact_matches += 1

        result_record = {
            "index": idx,
            "ground_truth": ground_truth_xml,
            "prediction": generated_xml_raw,
            "true_turns": true_turns,
            "predicted_turns": pred_turns,
            "true_turn_count": len(true_turns),
            "predicted_turn_count": len(pred_turns),
            "turn_count_exact": len(true_turns) == len(pred_turns),
            "paragraph_exact_match": true_turns == pred_turns,
        }
        with open(output_path, "a", encoding="utf-8") as output_handle:
            output_handle.write(json.dumps(result_record, ensure_ascii=False) + "\n")

        # Populate the metric arrays and penalize missing or hallucinated turns.
        max_len = max(len(true_turns), len(pred_turns))
        
        for i in range(max_len):
            t_s, t_a, t_b = true_turns[i] if i < len(true_turns) else ("Missing", "Missing", "no")
            p_s, p_a, p_b = pred_turns[i] if i < len(pred_turns) else ("Missing", "Missing", "no")

            y_true_speaker.append(t_s)
            y_pred_speaker.append(p_s)

            y_true_addressee.append(t_a)
            y_pred_addressee.append(p_a)

            y_true_bystanders.append(t_b)
            y_pred_bystanders.append(p_b)

        if args.verbose:
            # Show full details only for the first three paragraphs to keep logs compact.
            if idx < 3:
                print(f"\n[SAMPLE INSPECTION {idx + 1}/3]")
                print(f"  Target Input (Raw):   {input_messages[-1]['content'].split('Target paragraph to parse:')[-1].strip()[:90]}...")
                print(f"  Model Generated XML:  {generated_xml_raw}")
                print(f"  Ground Truth XML:     {ground_truth_xml}")
                print(f"  Turns found (predicted/true): {len(pred_turns)} / {len(true_turns)}")
                print("-" * 60)
        if (idx + 1) % 25 == 0 or (idx + 1) == len(eval_set):
            print(f"Evaluation progress: {idx + 1}/{len(eval_set)} paragraphs processed...")

    print("\n==================================================")
    print(f" METRICS REPORT FOR: {args.model_id}")
    print("==================================================")
    evaluated_paragraphs = len(eval_set)
    paragraph_exact_rate = (
        paragraph_exact_matches / evaluated_paragraphs
        if evaluated_paragraphs else 0.0
    )
    turn_count_exact_rate = (
        turn_count_exact_matches / evaluated_paragraphs
        if evaluated_paragraphs else 0.0
    )

    print("\n[STRUCTURAL METRICS]")
    print(f"Evaluated paragraphs:              {evaluated_paragraphs}")
    print(
        f"Exactly correct paragraphs:        {paragraph_exact_matches} "
        f"({paragraph_exact_rate:.2%})"
    )
    print(
        f"Exactly correct turn count:        {turn_count_exact_matches} "
        f"({turn_count_exact_rate:.2%})"
    )
    print(f"Ground-truth turns:                {total_true_turns}")
    print(f"Predicted turns:                   {total_pred_turns}")
    print(f"Missing turns:                     {total_missing_turns}")
    print(f"Extra or hallucinated turns:       {total_extra_turns}")
    print(f"Detailed results file:             {output_path}")

    # Use zero_division=0 for classes with no predicted or ground-truth samples.
    print("\n[LAYERS 1 & 2] SPEAKER EVALUATION:")
    print(classification_report(y_true_speaker, y_pred_speaker, zero_division=0))
    print("\n[LAYER 3] ADDRESSEE EVALUATION:")
    print(classification_report(y_true_addressee, y_pred_addressee, zero_division=0))
    print("\n[LAYER 4] BYSTANDERS (YES/NO) EVALUATION:")
    print(classification_report(y_true_bystanders, y_pred_bystanders, zero_division=0))
    print("==================================================")

if __name__ == "__main__":
    main()


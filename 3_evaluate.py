import os
import re
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import classification_report

def extract_xml_attributes(xml_string):
    """
    Hulpfunctie om met behulp van regex de geannoteerde lagen 
    uit de XML-structuur te extraheren.
    """
    if not isinstance(xml_string, str):
        return "Unknown", "Unknown", "no"
        
    s_match = re.search(r'S=["\']([^"\']*)["\']', xml_string)
    a_match = re.search(r'A=["\']([^"\']*)["\']', xml_string)
    b_match = re.search(r'bystanders=["\']([^"\']*)["\']', xml_string)
    
    speaker = s_match.group(1) if s_match else "Unknown"
    addressee = a_match.group(1) if a_match else "Unknown"
    bystanders = b_match.group(1) if b_match else "no"
    
    return speaker, addressee, bystanders

def main():
    # --- DYNAMISCHE PARAMETERS ---
    parser = argparse.ArgumentParser(description="Universele Evaluatie Engine voor Causal LM Modellen met LoRA")
    parser.add_argument(
        "--model_id", 
        type=str, 
        required=True, 
        help="Hugging Face model ID of lokaal pad naar het basismodel (bijv. meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--adapter_dir", 
        type=str, 
        required=True, 
        help="Het exacte (lokale) pad naar de map met bewaarde LoRA adapters"
    )
    parser.add_argument(
        "--data_file", 
        type=str, 
        default="intermediate_output/val.jsonl", 
        help="Pad naar het JSONL validatie/test-bestand"
    )
    args = parser.parse_args()

    print("==================================================")
    print(" UNIVERSALE EVALUATIE ENGINE INITIALISATIE")
    print("==================================================")
    print(f"--> Basismodel:      {args.model_id}")
    print(f"--> LoRA Adapters:   {args.adapter_dir}")
    print(f"--> Evaluatie Data:  {args.data_file}")
    print("==================================================")
    
    # --- DATA INLADEN ---
    dataset = load_dataset("json", data_files={"validation": args.data_file})
    
    # --- TOKENIZER & MODEL LADEN ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # device_map="auto" zorgt dat 8B/7B op 1 GPU past, en 70B/67B automatisch wordt geshard over alle actieve GPU's
    base_model_eval = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    
    eval_model = PeftModel.from_pretrained(base_model_eval, args.adapter_dir)
    eval_model.eval()

    y_true_speaker, y_pred_speaker = [], []
    y_true_addressee, y_pred_addressee = [], []
    y_true_bystanders, y_pred_bystanders = [], []

    eval_set = dataset["validation"]
    print("\n--- Start generatie en scoring op de dataset ---")

    for idx, sample in enumerate(eval_set):
        messages = sample["messages"]
        input_messages = messages[:-1]
        ground_truth_xml = messages[-1]["content"]
    
        # apply_chat_template werkt universeel mits het model een correcte chat_template in de config heeft (zoals Llama, Qwen en DeepSeek)
        prompt = tokenizer.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = eval_model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs.sequences[0][input_length:]
        
        token_probabilities = []
        for i, token_id in enumerate(generated_tokens):
            if token_id == tokenizer.eos_token_id:
                break
            logits = outputs.scores[i][0]
            probs = torch.softmax(logits, dim=-1)
            token_prob = probs[token_id].item()
            token_probabilities.append(token_prob)

        avg_confidence = sum(token_probabilities) / len(token_probabilities) if token_probabilities else 0.0

        generated_xml_raw = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_xml = generated_xml_raw.replace("<quote", f'<quote confidence="{avg_confidence:.4f}"')

        true_s, true_a, true_b = extract_xml_attributes(ground_truth_xml)
        pred_s, pred_a, pred_b = extract_xml_attributes(generated_xml)

        y_true_speaker.append(true_s)
        y_pred_speaker.append(pred_s)

        y_true_addressee.append(true_a)
        y_pred_addressee.append(pred_a)

        y_true_bystanders.append(true_b)
        y_pred_bystanders.append(pred_b)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(eval_set):
            print(f"Geëvalueerd: {idx + 1}/{len(eval_set)} turns...")

    # --- PERFORMANCE RAPPORT ---
    print("\n==================================================")
    print(f" METRIEKEN RAPPORT VOOR: {args.model_id}")
    print("==================================================")
    print("\n[LAAG 1 & 2] SPREKER (SPEAKER) EVALUATIE:")
    print(classification_report(y_true_speaker, y_pred_speaker, zero_division=0))
    print("\n[LAAG 3] GEADRESSEERDE (ADDRESSEE) EVALUATIE:")
    print(classification_report(y_true_addressee, y_pred_addressee, zero_division=0))
    print("\n[LAAG 4] BYSTANDERS (YES/NO) EVALUATIE:")
    print(classification_report(y_true_bystanders, y_pred_bystanders, zero_division=0))
    print("==================================================")

if __name__ == "__main__":
    main()

import os
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
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
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    
    # Inladen van de geprepareerde datasets
    dataset = load_dataset("json", data_files={
        "train": "intermediate_output/train.jsonl", 
        "validation": "intermediate_output/val.jsonl"
    })
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # LoRA configuratie voor tekstanalyse en structuur-leren
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    scratch_project_dir = os.path.join(os.environ.get("VSC_SCRATCH", "."), "JaneAusten")
    output_model_dir = os.path.join(scratch_project_dir, "fine_tuned_austen")

    training_args = SFTConfig(
        output_dir=os.path.join(scratch_project_dir, "results_austen_70b"),
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        weight_decay=0.01,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=5,
        dataloader_num_workers=4,
        report_to="none",
        max_length=2048,
        gradient_checkpointing=True,
        deepspeed="deepspeed_config.json",
    )
    
    trainer = SFTTrainer(
        model=model_id,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer
    )

    print("Start de training via DeepSpeed ZeRO-3...")
    checkpoint_dir = os.path.join(scratch_project_dir, "results_austen_70b")
    resume_from_checkpoint = False
    
    # Als de map bestaat en gevuld is met eerdere checkpoints, pakken we die op
    if os.path.exists(checkpoint_dir) and any("checkpoint" in d for d in os.listdir(checkpoint_dir)):
        resume_from_checkpoint = True
        print("--> Eerder checkpoint gevonden! Training zal automatisch herstarten vanaf het laatste punt.")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Sla zowel het model als de tokenizer op in de scratch-omgeving
    trainer.model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"Model en tokenizer succesvol opgeslagen in: {output_model_dir}")

    # ========================================================================
    # EVALUATIE FASE: PRECISION & RECALL PER LAAG BEREKENEN
    # ========================================================================
    if trainer.is_main_process:
       print("\n--- Start Evaluatie op de Testset (Validation Split) ---")
       
       # Zet het getrainde model in evaluatie-modus
       trainer.model.eval()
    
       y_true_speaker, y_pred_speaker = [], []
       y_true_addressee, y_pred_addressee = [], []
       y_true_bystanders, y_pred_bystanders = [], []
    
       eval_set = dataset["validation"]
    
       # We lopen door de test-samples heen
       for idx, sample in enumerate(eval_set):
           messages = sample["messages"]
        
           # Scheid de input (system + user) van het verwachte antwoord (assistant)
           input_messages = messages[:-1]
           ground_truth_xml = messages[-1]["content"]
        
           # Format de input volgens de Llama-3 chat template
           prompt = tokenizer.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True)
           inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
           # Genereer de XML output (greedy decoding voor maximale stabiliteit)
           with torch.no_grad():
               outputs = trainer.model.generate(
                   **inputs, 
                   max_new_tokens=512, 
                   do_sample=False,
                   pad_token_id=tokenizer.eos_token_id
            )
            
           generated_xml = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
           # Extraheer de attributen uit zowel de grondwaarheid als de voorspelling
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

       # Bereken en toon de statistieken per laag
       print("\n==================================================")
       print(" PERFORMANCE RAPPORT PER ANNOTATIELAAG")
       print("==================================================")
    
       # Laag 1 & 2: Spreker (Multiclass)
       print("\n[LAAG 1 & 2] SPREKER (SPEAKER) EVALUATIE:")
       # zero_division=0 voorkomt crashes als een personage nooit correct is geraden
       print(classification_report(y_true_speaker, y_pred_speaker, zero_division=0))
    
       # Laag 3: Geadresseerde (Multiclass)
       print("\n[LAAG 3] GEADRESSEERDE (ADDRESSEE) EVALUATIE:")
       print(classification_report(y_true_addressee, y_pred_addressee, zero_division=0))
    
       # Laag 4: Aanwezigheid van Bystanders (Binair / Categorisch)
       print("\n[LAAG 4] BYSTANDERS (YES/NO) EVALUATIE:")
       print(classification_report(y_true_bystanders, y_pred_bystanders, zero_division=0))
       print("==================================================")

if __name__ == "__main__":
    main()

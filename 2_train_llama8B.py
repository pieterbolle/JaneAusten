import os
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from sklearn.metrics import classification_report

def extract_all_turns_metadata(xml_string):
    """
    Oogst de metadata van ALLE turns binnen een paragraaf om 
    multi-turn alinea's correct te kunnen beoordelen.
    """
    if not isinstance(xml_string, str):
        return []
        
    # Vind alle turn-tags binnen de paragraaf
    turn_matches = re.findall(r'<turn\s+(.*?)>', xml_string)
    turns_metadata = []
    
    for meta in turn_matches:
        s_match = re.search(r'S=["\']([^"\']*)["\']', meta)
        a_match = re.search(r'A=["\']([^"\']*)["\']', meta)
        b_match = re.search(r'bystanders=["\']([^"\']*)["\']', meta)
        
        speaker = s_match.group(1) if s_match else "Unknown"
        addressee = a_match.group(1) if a_match else "Unknown"
        bystanders = b_match.group(1) if b_match else "no"
        turns_metadata.append((speaker, addressee, bystanders))
        
    return turns_metadata

def main():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    dataset = load_dataset("json", data_files={
        "train": "intermediate_output/train.jsonl", 
        "validation": "intermediate_output/val.jsonl"
    })
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    def tokenize_function(examples):
        texts = [tokenizer.apply_chat_template(messages, tokenize=False) for messages in examples["messages"]]
        result = tokenizer(texts, truncation=True, max_length=2048, padding="max_length")
        result["labels"] = result["input_ids"].copy()
        return result

    print("Pre-tokenizing datasets...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    scratch_project_dir = os.path.join(os.environ.get("VSC_SCRATCH", "."), "JaneAusten")
    output_model_dir = os.path.join(scratch_project_dir, "fine_tuned_austen_8b")

    training_args = TrainingArguments(
        output_dir=os.path.join(scratch_project_dir, "results_austen_8b"),
        eval_strategy="steps",
        eval_steps=40,
        save_strategy="steps",
        save_steps=40,
        save_total_limit=2,
        learning_rate=2e-4,              
        per_device_train_batch_size=4,   
        per_device_eval_batch_size=4,    
        gradient_accumulation_steps=4,   
        weight_decay=0.01,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=5,
        dataloader_num_workers=2,        
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    base_model.config.use_cache = False
    model = get_peft_model(base_model, peft_config)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    print("Start de training van het 8B model via native DDP...")
    checkpoint_dir = os.path.join(scratch_project_dir, "results_austen_8b")
    resume_from_checkpoint = False
    
    if os.path.exists(checkpoint_dir) and any("checkpoint" in d for d in os.listdir(checkpoint_dir)):
        resume_from_checkpoint = True
        print("--> Eerder checkpoint gevonden! Training start vanaf het laatste punt.")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    trainer.model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"Model en tokenizer succesvol opgeslagen in: {output_model_dir}")

    # Zorg dat de Llama 3 stop criteria voor de turn grens geladen worden
    stop_tokens = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int):
        stop_tokens.append(eot_id)

    if trainer.is_world_process_zero():
       print("\n--- Start Evaluatie op Paragraaf-niveau ---")

       del model
       del trainer
       torch.cuda.empty_cache()

       from peft import PeftModel

       base_model_eval = AutoModelForCausalLM.from_pretrained(
           model_id,
           torch_dtype=torch.bfloat16,
           device_map="auto",
           attn_implementation="flash_attention_2"
       )
       eval_model = PeftModel.from_pretrained(base_model_eval, output_model_dir)
       eval_model.eval()

       y_true_speaker, y_pred_speaker = [], []
       y_true_addressee, y_pred_addressee = [], []
       y_true_bystanders, y_pred_bystanders = [], []

       eval_set = dataset["validation"]

       for idx, sample in enumerate(eval_set):
           messages = sample["messages"]
           input_messages = messages[:-1]
           ground_truth_xml = messages[-1]["content"]
        
           prompt = tokenizer.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True)
           inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

           with torch.no_grad():
               outputs = eval_model.generate(
                   **inputs, 
                   max_new_tokens=512, 
                   do_sample=False,
                   pad_token_id=tokenizer.eos_token_id,
                   eos_token_id=stop_tokens,
                   return_dict_in_generate=True,
                   output_scores=True
               )
            
           input_length = inputs["input_ids"].shape[1]
           generated_tokens = outputs.sequences[0][input_length:]
           
           generated_xml = tokenizer.decode(generated_tokens, skip_special_tokens=True)

           # Multi-turn extractie voor de matrix
           true_turns = extract_all_turns_metadata(ground_truth_xml)
           pred_turns = extract_all_turns_metadata(generated_xml)

           # Lijstlengtes rechttrekken om uitlijningsfouten (bijv. model mist een tag) af te straffen
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

           if (idx + 1) % 40 == 0 or (idx + 1) == len(eval_set):
               print(f"Geëvalueerd: {idx + 1}/{len(eval_set)} paragrafen...")

       print("\n==================================================")
       print(" PERFORMANCE RAPPORT (PARAGRAAF-NIVEAU)")
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

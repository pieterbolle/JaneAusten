import os
import re
import torch
import transformers.trainer
import transformers.utils.import_utils
transformers.trainer.check_torch_load_is_safe = lambda *args, **kwargs: None
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from sklearn.metrics import classification_report
import deepspeed

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
    output_model_dir = os.path.join(scratch_project_dir, "fine_tuned_austen")

    # Swappen van SFTConfig naar de stabiele TrainingArguments
    training_args = TrainingArguments(
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
        dataloader_num_workers=0,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        deepspeed="deepspeed_config.json",
        ddp_timeout=7200
    )

    ds_config_path = "deepspeed_config.json"

    # Laad en shard het model én de adapters direct correct via DeepSpeed
    with deepspeed.zero.Init(config_dict_or_path=ds_config_path):
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        base_model.config.use_cache = False
        model = get_peft_model(base_model, peft_config)
    
    # Gebruik de standaard, stabiele Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    print("Start de training via DeepSpeed ZeRO-3...")
    checkpoint_dir = os.path.join(scratch_project_dir, "results_austen_70b")
    resume_from_checkpoint = False
    
    if os.path.exists(checkpoint_dir) and any("checkpoint" in d for d in os.listdir(checkpoint_dir)):
        resume_from_checkpoint = True
        print("--> Eerder checkpoint gevonden! Training zal automatisch herstarten vanaf het laatste punt.")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"Model en tokenizer succesvol opgeslagen in: {output_model_dir}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # ========================================================================
    # EVALUATIE FASE
    # ========================================================================
    if training_args.should_save:
       print("\n--- Start Evaluatie op de Testset (Validation Split) ---")

       del model
       del trainer
       torch.cuda.empty_cache()

       from peft import PeftModel

       print("--> Laden van het samengevoegde model voor evaluatie...")
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

       # We gebruiken de originele dataset (met de 'messages' kolom) voor de evaluatie loop
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

       print("\n==================================================")
       print(" PERFORMANCE RAPPORT PER ANNOTATIELAAG")
       print("==================================================")

       print("\n[LAAG 1 & 2] SPREKER (SPEAKER) EVALUATIE:")
       print(classification_report(y_true_speaker, y_pred_speaker, zero_division=0))

       print("\n[LAAG 3] GEADRESSEERDE (ADDRESSEE) EVALUATIE:")
       print(classification_report(y_true_addressee, y_pred_addressee, zero_division=0))

       print("\n[LAAG 4] BYSTANDERS (YES/NO) EVALUATIE:")
       print(classification_report(y_true_bystanders, y_pred_bystanders, zero_division=0))
       print("==================================================")

    else:
       import time
       torch.distributed.barrier()

if __name__ == "__main__":
    main()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re

def load_inference_model(base_model_id, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tokenizer

def segment_text_to_turns(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Eenvoudige segmentatie op basis van alinea's (double newlines)
    # Voor een productieomgeving kan dit verfijnd worden naar zinssegmentatie
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

def run_inference(model, tokenizer, paragraphs):
    system_prompt = (
        "You are an expert NLP system. Convert the raw narrative literary text into a structured XML turn format. "
        "Identify the Speaker (S), Addressee (A), Bystanders (yes/no), and wrap narrative interruptions within the turn in <interruption> tags."
    )
    
    output_xml = ["<root>"]
    context_history = []
    
    for idx, para in enumerate(paragraphs):
        context_str = "\n".join(context_history[-3:])
        user_input = f"Context history:\n{context_str}\n\nTarget text to parse:\n{para}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        output_xml.append(generated_text.strip())
        
        # Voeg de huidige ruwe tekst toe aan de context voor de volgende beurt
        context_history.append(para)
        
        if idx % 50 == 0:
            print(f"Voortgang: {idx}/{len(paragraphs)} alinea's verwerkt.")
            
    output_xml.append("</root>")
    return "\n".join(output_xml)

if __name__ == "__main__":
    BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
    ADAPTER_DIR = os.path.join(scratch_project_dir, "fine_tuned_austen")
    
    print("Model inladen...")
    model, tokenizer = load_inference_model(BASE_MODEL, ADAPTER_DIR)
    
    # Resterend oeuvre verwerken
    print("Tekst verwerken...")
    paragraphs = segment_text_to_turns("sense_and_sensibility.txt")
    xml_result = run_inference(model, tokenizer, paragraphs)
    
    with open("sense_and_sensibility_annotated.xml", "w", encoding="utf-8") as f:
        f.write(xml_result)
        
    print("Annotatie compleet! Bestand opgeslagen als 'sense_and_sensibility_annotated.xml'")

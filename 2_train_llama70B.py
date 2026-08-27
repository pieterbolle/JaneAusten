import argparse
import json
import math
import os
from pathlib import Path

import torch
import time
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        )
from transformers.trainer_utils import get_last_checkpoint

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MAX_LENGTH = 2048
RUN_NAME = "austen_70b_tokenfix_v5"
DEEPSPEED_CONFIG = "deepspeed_config.json"
TRAIN_FILE = "intermediate_output/train.jsonl"
VALIDATION_FILE = "intermediate_output/val.jsonl"
LONG_EXAMPLES_AUDIT_FILE = "intermediate_output/long_examples_audit.json"
SHORT_TARGET_THRESHOLD = 10
ALLOW_CHECKPOINT_RESUME = False


def parse_args():
    """Parse command-line options for CPU validation and GPU training."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--preprocess-only",
            action="store_true",
            help="Run CPU-safe preprocessing tests and exit before model loading.",
            )
    parser.add_argument(
            "--refresh-cache",
            action="store_true",
            help="Ignore cached Hugging Face filter and map results.",
            )
    parser.add_argument(
            "--debug-samples",
            type=int,
            default=3,
            help="Number of decoded training targets to print.",
            )
    parser.add_argument(
            "--smoke-test",
            action="store_true",
            help="Run two GPU optimizer steps on small dataset subsets.",
            )
    return parser.parse_args()


def to_token_id_list(value):
    """Normalize tokenizer output to a flat Python list of integer token IDs."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if hasattr(value, "ids"):
        value = value.ids
    if isinstance(value, dict) and "input_ids" in value:
        value = value["input_ids"]
    if hasattr(value, "input_ids"):
        value = value.input_ids
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError(f"Unsupported tokenizer output: {type(value).__name__}")

    result = []
    for position, token_id in enumerate(value):
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        if not isinstance(token_id, int):
            raise TypeError(
                    f"Token {position} has type {type(token_id).__name__}, expected int."
                    )
        result.append(token_id)
    return result


def apply_chat_template_ids(tokenizer, messages, add_generation_prompt):
    """Apply the chat template and always return integer token IDs."""
    return to_token_id_list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                )
            )


def find_last_subsequence(sequence, subsequence):
    """Return the start index of the last exact subsequence match."""
    if not subsequence or len(subsequence) > len(sequence):
        return None
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start:start + len(subsequence)] == subsequence:
            return start
    return None


def build_assistant_header_ids(tokenizer):
    """Build the Llama assistant-header tokens without hard-coded IDs."""
    text = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ids = to_token_id_list(tokenizer.encode(text, add_special_tokens=False))
    if not ids:
        raise RuntimeError("The assistant header produced no token IDs.")
    return ids


def audit_and_filter_dataset(dataset, tokenizer, refresh_cache, is_main_process):
    """Audit excluded rows, write an audit file, and filter invalid or long rows."""
    audit_rows = []
    for split in ("train", "validation"):
        for index, row in enumerate(dataset[split]):
            messages = row["messages"]
            if not messages:
                audit_rows.append({"split": split, "index": index, "reason": "empty_messages"})
                continue
            if messages[-1].get("role") != "assistant":
                audit_rows.append(
                        {
                            "split": split,
                            "index": index,
                            "reason": "final_role_is_not_assistant",
                            "final_role": messages[-1].get("role"),
                            }
                        )
                continue
            length = len(apply_chat_template_ids(tokenizer, messages, False))
            if length > MAX_LENGTH:
                audit_rows.append(
                        {
                            "split": split,
                            "index": index,
                            "reason": "sequence_too_long",
                            "token_length": length,
                            "excess_tokens": length - MAX_LENGTH,
                            }
                        )

    if is_main_process:
        audit_path = Path(LONG_EXAMPLES_AUDIT_FILE)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
                json.dumps(audit_rows, indent=2, ensure_ascii=False),
                encoding="utf-8",
                )
        print(f"Pre-filter audit found {len(audit_rows)} excluded examples.")
        print(f"Audit written to: {audit_path}")
        for item in audit_rows[:20]:
            print(f"Excluded example: {item}")

    def keep_example(example):
        messages = example["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            return False
        return len(apply_chat_template_ids(tokenizer, messages, False)) <= MAX_LENGTH

    original_sizes = {split: len(dataset[split]) for split in dataset}
    filtered = dataset.filter(
            keep_example,
            batched=False,
            desc="Filtering long conversations",
            load_from_cache_file=not refresh_cache,
            )
    for split in ("train", "validation"):
        removed = original_sizes[split] - len(filtered[split])
        if is_main_process:
            print(
                f"{split}: retained {len(filtered[split])} of "
                f"{original_sizes[split]} examples; removed {removed}.",
                flush=True,
            )
        if len(filtered[split]) == 0:
            raise RuntimeError(f"The {split} split is empty after filtering.")
    return filtered


def make_tokenize_function(tokenizer, assistant_header_ids):
    """Create labels that apply loss only to the final assistant answer."""
    def tokenize_function(examples, indices):
        output = {"input_ids": [], "attention_mask": [], "labels": []}
        for index, messages in zip(indices, examples["messages"]):
            full_ids = apply_chat_template_ids(tokenizer, messages, False)
            header_start = find_last_subsequence(full_ids, assistant_header_ids)
            if header_start is None:
                tail = tokenizer.decode(
                        full_ids[-100:],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                        )
                raise ValueError(
                        f"Example {index}: assistant header not found. Tail: {tail!r}"
                        )
            response_start = header_start + len(assistant_header_ids)
            if response_start >= len(full_ids):
                raise ValueError(f"Example {index}: assistant target is empty.")
            labels = [-100] * response_start + full_ids[response_start:]
            attention_mask = [1] * len(full_ids)
            if not (len(full_ids) == len(attention_mask) == len(labels)):
                raise RuntimeError(f"Example {index}: token field lengths differ.")
            output["input_ids"].append(full_ids)
            output["attention_mask"].append(attention_mask)
            output["labels"].append(labels)
        return output
    return tokenize_function


def validate_dataset(
    dataset,
    tokenized,
    tokenizer,
    debug_samples,
    is_main_process,
):
    """Validate cardinality, masking, lengths, short targets, and decoding."""
    short_targets = []
    split_statistics = {}

    for split in ("train", "validation"):
        if len(dataset[split]) != len(tokenized[split]):
            raise RuntimeError(
                f"{split}: dataset cardinality changed during tokenization."
            )

        min_length = None
        min_targets = None
        max_length = 0
        max_targets = 0

        for index, row in enumerate(tokenized[split]):
            input_ids = to_token_id_list(row["input_ids"])
            attention_mask = to_token_id_list(row["attention_mask"])
            labels = to_token_id_list(row["labels"])

            if not (len(input_ids) == len(attention_mask) == len(labels)):
                raise RuntimeError(
                    f"{split}[{index}]: token field lengths differ."
                )

            target_ids = [
                token_id
                for token_id, label in zip(input_ids, labels)
                if label != -100
            ]
            if not target_ids:
                raise RuntimeError(
                    f"{split}[{index}]: no active target tokens."
                )

            sequence_length = len(input_ids)
            target_length = len(target_ids)
            min_length = (
                sequence_length
                if min_length is None
                else min(min_length, sequence_length)
            )
            min_targets = (
                target_length
                if min_targets is None
                else min(min_targets, target_length)
            )
            max_length = max(max_length, sequence_length)
            max_targets = max(max_targets, target_length)

            if target_length < SHORT_TARGET_THRESHOLD:
                short_targets.append(
                    {
                        "split": split,
                        "index": index,
                        "target_tokens": target_length,
                        "target": tokenizer.decode(
                            target_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                    }
                )

        split_statistics[split] = {
            "count": len(tokenized[split]),
            "min_length": min_length,
            "max_length": max_length,
            "min_targets": min_targets,
            "max_targets": max_targets,
        }

    if not is_main_process:
        return

    print("\n" + "=" * 72)
    print("TOKENIZED DATASET VALIDATION")
    print("=" * 72)

    for split in ("train", "validation"):
        stats = split_statistics[split]
        print(
            f"{split}: raw={len(dataset[split])}, "
            f"tokenized={stats['count']}"
        )
        print(
            f"{split}: sequence length range="
            f"{stats['min_length']}..{stats['max_length']}; "
            f"active target range="
            f"{stats['min_targets']}..{stats['max_targets']}"
        )

    print(
        f"Targets shorter than {SHORT_TARGET_THRESHOLD} tokens: "
        f"{len(short_targets)}"
    )
    for item in short_targets[:20]:
        print(
            f"Short target {item['split']}[{item['index']}], "
            f"{item['target_tokens']} tokens: {item['target']!r}"
        )

    for index in range(min(debug_samples, len(tokenized["train"]))):
        row = tokenized["train"][index]
        input_ids = to_token_id_list(row["input_ids"])
        labels = to_token_id_list(row["labels"])
        target_ids = [
            token_id
            for token_id, label in zip(input_ids, labels)
            if label != -100
        ]
        print(f"\nDecoded training target {index}:")
        print(
            tokenizer.decode(
                target_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[:1500]
        )

    print("=" * 72 + "\n", flush=True)


def validate_collator(tokenized, collator, is_main_process):
    """Test dynamic padding and label masking on CPU."""
    features = [tokenized["train"][i] for i in range(min(4, len(tokenized["train"])))]
    batch = collator(features)
    if batch["input_ids"].shape != batch["attention_mask"].shape:
        raise RuntimeError("Collated input and attention shapes differ.")
    if batch["input_ids"].shape != batch["labels"].shape:
        raise RuntimeError("Collated input and label shapes differ.")
    padding = batch["attention_mask"] == 0
    if padding.any() and not torch.all(batch["labels"][padding] == -100):
        raise RuntimeError("Padded label positions are not masked with -100.")
    active = (batch["labels"] != -100).sum(dim=1)
    if torch.any(active == 0):
        raise RuntimeError("A collated sample has no active target tokens.")
    if is_main_process:
        print("CPU collator test passed.")
        print(f"Collated batch shape: {tuple(batch['input_ids'].shape)}")
        print(f"Active target tokens per sample: {active.tolist()}")


def print_training_geometry(args, train_size):
    """Print the expected effective batch size and optimizer-step count."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(train_size / global_batch)
    total_steps = math.ceil(steps_per_epoch * float(args.num_train_epochs))
    print(f"WORLD_SIZE: {world_size}")
    print(f"Effective global batch size: {global_batch}")
    print(f"Expected optimizer steps per epoch: {steps_per_epoch}")
    print(f"Expected total optimizer steps: {total_steps}")
    if total_steps <= 3:
        raise RuntimeError("The expected optimizer-step count is unrealistically low.")


def main():
    rank = int(os.environ.get("RANK", "0"))
    is_main_process = rank == 0
    cli = parse_args()
    gpu_available = torch.cuda.is_available()
    preprocess_only = cli.preprocess_only or not gpu_available
    if is_main_process:
        print(f"CUDA available: {gpu_available}")
        if not gpu_available:
            print(
                    "CPU mode detected. The script will run preprocessing, dataset, "
                    "decoding, and collator tests, then exit before loading the 70B model."
                    )

    for file_name in (TRAIN_FILE, VALIDATION_FILE):
        if not Path(file_name).is_file():
            raise FileNotFoundError(f"Dataset file not found: {file_name}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.eos_token is None:
        raise RuntimeError("The tokenizer does not define an EOS token.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    assistant_header_ids = build_assistant_header_ids(tokenizer)
    if is_main_process:
        print(f"Assistant header token IDs: {assistant_header_ids}")

    dataset = load_dataset(
            "json",
            data_files={"train": TRAIN_FILE, "validation": VALIDATION_FILE},
            )
    dataset = audit_and_filter_dataset(dataset, tokenizer, cli.refresh_cache, is_main_process)

    if is_main_process:
        print("Tokenizing conversations...")
    tokenized = dataset.map(
            make_tokenize_function(tokenizer, assistant_header_ids),
            batched=True,
            batch_size=100,
            with_indices=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing conversations",
            load_from_cache_file=not cli.refresh_cache,
            )

    if is_main_process:
        print("Creating DataCollatorForSeq2Seq...")
    collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
            return_tensors="pt",
            )
    validate_dataset(
        dataset,
        tokenized,
        tokenizer,
        cli.debug_samples,
        is_main_process,
    )
    validate_collator(tokenized, collator, is_main_process)

    if preprocess_only:
        if is_main_process:
            print(
                "All CPU-safe preprocessing tests passed. "
                "Exiting before model loading.",
                flush=True,
            )
        return

    if cli.smoke_test:
        tokenized["train"] = tokenized["train"].select(range(min(32, len(tokenized["train"]))))
        tokenized["validation"] = tokenized["validation"].select(
                range(min(8, len(tokenized["validation"])))
                )
        if is_main_process:
            print(
                    f"GPU smoke test enabled: train={len(tokenized['train'])}, "
                    f"validation={len(tokenized['validation'])}, max_steps=2."
                    )

    if not Path(DEEPSPEED_CONFIG).is_file():
        raise FileNotFoundError(f"DeepSpeed configuration not found: {DEEPSPEED_CONFIG}")
    scratch = Path(os.environ.get("VSC_SCRATCH", ".")) / "JaneAusten"
    suffix = f"{RUN_NAME}_smoke" if cli.smoke_test else RUN_NAME
    results_dir = scratch / f"results_{suffix}"
    model_dir = scratch / f"fine_tuned_{suffix}"

    if cli.smoke_test:
        eval_strategy = "no"
        save_strategy = "no"
        accumulation = 2
        epochs = 1
        max_steps = 2
        logging_steps = 1
    else:
        eval_strategy = "steps"
        save_strategy = "steps"
        accumulation = 16
        epochs = 3
        max_steps = -1
        logging_steps = 5

    training_args = TrainingArguments(
            output_dir=str(results_dir),
            eval_strategy=eval_strategy,
            eval_steps=40,
            save_strategy=save_strategy,
            save_steps=40,
            save_total_limit=2,
            learning_rate=1e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=accumulation,
            weight_decay=0.01,
            num_train_epochs=epochs,
            max_steps=max_steps,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=logging_steps,
            dataloader_num_workers=0,
            report_to="none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            deepspeed=DEEPSPEED_CONFIG,
            ddp_timeout=7200,
            remove_unused_columns=False,
            )

    peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            )

    if is_main_process:
        print(
            "Starting ZeRO-3-aware model loading through "
            "the Transformers Trainer integration.",
            flush=True,
        )

    model_load_start = time.perf_counter()

    print(
        f"[rank {rank}] entering from_pretrained",
        flush=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    print(
        f"[rank {rank}] completed from_pretrained",
        flush=True,
    )

    base_model.config.use_cache = False

    if is_main_process:
        base_model_seconds = time.perf_counter() - model_load_start
        print(
            f"Base model loaded in {base_model_seconds / 60:.2f} minutes. "
            "Injecting LoRA adapters.",
            flush=True,
        )

    lora_start = time.perf_counter()

    model = get_peft_model(
        base_model,
        peft_config,
        autocast_adapter_dtype=False,
    )

    trainable_dtype_counts = {}
    partition_dtype_counts = {}

    for _, parameter in model.named_parameters():
        if parameter.requires_grad:
            dtype_name = str(parameter.dtype)
            trainable_dtype_counts[dtype_name] = (
                trainable_dtype_counts.get(dtype_name, 0)
                + parameter.numel()
            )
            ds_tensor = getattr(parameter, "ds_tensor", None)
            if ds_tensor is not None:
                partition_dtype = str(ds_tensor.dtype)
                partition_dtype_counts[partition_dtype] = (
                     partition_dtype_counts.get(partition_dtype, 0)
                     + ds_tensor.numel()
                )  

    print(
        f"[rank {rank}] trainable parameter dtypes: "
        f"{trainable_dtype_counts}",
        flush=True,
    )

    if partition_dtype_counts:
        print(
            f"[rank {rank}] trainable DeepSpeed partition dtypes: "
            f"{partition_dtype_counts}",
            flush=True,
        )
    else:
        print(
            f"[rank {rank}] no DeepSpeed ds_tensor attributes were found "
            "on trainable parameters before Trainer initialization.",
            flush=True,
        )

    if not trainable_dtype_counts:
        raise RuntimeError(
             "No trainable LoRA parameters were found after adapter injection."
        )

    expected_dtype = str(torch.bfloat16)

    unexpected_parameter_dtypes = {
        dtype_name: parameter_count
        for dtype_name, parameter_count in trainable_dtype_counts.items()
        if dtype_name != expected_dtype
    }

    unexpected_partition_dtypes = {
        dtype_name: parameter_count
        for dtype_name, parameter_count in partition_dtype_counts.items()
        if dtype_name != expected_dtype
    }

    if unexpected_parameter_dtypes or unexpected_partition_dtypes:
        raise RuntimeError(
            "LoRA parameters do not all use bfloat16. "
            f"Unexpected parameter dtypes: {unexpected_parameter_dtypes}; "
            f"unexpected DeepSpeed partition dtypes: "
            f"{unexpected_partition_dtypes}."
        )

    print(
        f"[rank {rank}] completed LoRA injection",
        flush=True,
    )
    
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if is_main_process:
        lora_seconds = time.perf_counter() - lora_start
        total_seconds = time.perf_counter() - model_load_start
        print(
            f"LoRA injection completed in {lora_seconds:.2f} seconds.",
            flush=True,
        )
        print(
            f"Total model and LoRA initialization time: "
            f"{total_seconds / 60:.2f} minutes.",
            flush=True,
        )
        print(
            "All ranks completed model and LoRA initialization.",
            flush=True,
        )

    model.enable_input_require_grads()

    if is_main_process:
        model.print_trainable_parameters()
        print("Creating Trainer.", flush=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
    )

    checkpoint = None
    if (
        ALLOW_CHECKPOINT_RESUME
        and not cli.smoke_test
        and results_dir.is_dir()
    ):
        checkpoint = get_last_checkpoint(str(results_dir))

    if is_main_process:
        if cli.smoke_test:
            print(
                "Expected smoke-test optimizer steps: 2",
                flush=True,
            )
        else:
            print_training_geometry(
                training_args,
                len(tokenized["train"]),
            )

        print(
            "Checkpoint resume: "
            f"{checkpoint if checkpoint else 'disabled'}",
            flush=True,
        )
        print("Starting trainer.train().", flush=True)

    training_start = time.perf_counter()
    trainer.train(resume_from_checkpoint=checkpoint)
    training_seconds = time.perf_counter() - training_start

    if is_main_process:
        print(
            f"trainer.train() completed in "
            f"{training_seconds / 60:.2f} minutes.",
            flush=True,
        )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if cli.smoke_test:
        if is_main_process:
            print(
                "GPU smoke test completed successfully. "
                "Skipping final model save.",
                flush=True,
            )
        return

    trainer.save_model(str(model_dir))

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(model_dir))
        print(
            f"Model adapter and tokenizer saved to: {model_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()


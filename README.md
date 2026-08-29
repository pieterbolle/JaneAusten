# Jane Austen Dialogue Annotation Pipeline

## 1. Purpose

This project fine-tunes a large language model to identify and annotate speech in nineteenth-century English prose. The current production pipeline uses **Meta Llama 3.3 70B Instruct** with LoRA adapters. It converts annotated source material into chat-style training examples, fine-tunes the model, evaluates it on held-out examples, applies it to complete Jane Austen works, and optionally normalizes speaker and addressee names afterwards.

The pipeline produces XML in which dialogue turns can contain:

- `S`: speaker;
- `A`: addressee;
- `bystanders`: whether other people are present;
- `indirectspeech`: whether the turn represents indirect speech;
- confidence fields for the turn and its attributes;
- nested `<interruption>` elements for narrative reporting clauses inside a continuous turn.

The 70B path is the maintained implementation. `2_train_llama8B.py` and its 8B SLURM jobs are retained only for historical reference and must be considered deprecated.

---

## 2. Maintained and deprecated components

### Maintained production path

1. `1A_convert2xml.py`, when starting from the original semicolon-separated manual annotation file.
2. `1_prepare_data.py`.
3. `2_train_llama70B.py` with `deepspeed_config.json`.
4. `3_evaluate.py`.
5. `4_inference.py`.
6. `5_normalize_jane_austen_xml.py` plus `jane_austen_character_normalization_v2.json`.

### Deprecated path

- `2_train_llama8B.py`;
- `janeausten_8B_job.sh`;
- `janeausten_eval_8B_job.sh`.

The 8B trainer predates corrections made in the 70B implementation. In particular, it trains on all chat tokens rather than masking the system and user portions, uses fixed-length padding, lacks the 70B data auditing and filtering safeguards, and contains older evaluation logic. It should not be used as a template for new experiments.

---

## 3. Directory layout

A practical project layout is:

```text
JaneAusten/
├── input/
│   ├── pride_and_prejudice_JK.csv
│   ├── emma.txt
│   ├── lady_susan.txt
│   ├── mansfield_park.txt
│   ├── northanger_abbey.txt
│   ├── persuasion.txt
│   └── sense_and_sensibility.txt
├── intermediate_output/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── long_examples_audit.json
├── output/
│   ├── pride_and_prejudice_clean.xml
│   ├── *_annotated_llama70B.xml
│   ├── *_annotated_llama70B_audit.jsonl
│   └── normalised/
├── 1A_convert2xml.py
├── 1_prepare_data.py
├── 2_train_llama70B.py
├── 3_evaluate.py
├── 4_inference.py
├── 5_normalize_jane_austen_xml.py
├── deepspeed_config.json
└── jane_austen_character_normalization_v2.json
```

Model checkpoints and Hugging Face caches are written under `$VSC_SCRATCH`, rather than the project directory.

---

## 4. Environment and security

### Required software

The supplied SLURM scripts were designed for the KU Leuven VSC WICE H100 environment and load:

```bash
module purge
module load cluster/wice/gpu
module load foss/2022b
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1
```

They then activate the project virtual environment:

```bash
cd /vsc-hard-mounts/leuven-data/381/vsc38154/JaneAusten/
source venv/bin/activate
```

Dependencies should be installed once when creating the virtual environment, not during every production job:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel setuptools ninja
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install flash_attn --no-build-isolation --no-cache-dir
```

### Hugging Face authentication

Do not store a Hugging Face access token directly in a tracked SLURM script. Use an environment variable supplied outside the repository, for example:

```bash
export HF_TOKEN="$(cat "$HOME/.secrets/huggingface_token")"
```

or authenticate interactively before submitting jobs. Any token that has already been committed or shared should be revoked and replaced.

Recommended cache settings are:

```bash
export HF_HOME="$VSC_SCRATCH/huggingface_cache"
export TRANSFORMERS_CACHE="$VSC_SCRATCH/huggingface_cache"
export TRITON_CACHE_DIR="$VSC_SCRATCH/triton_cache"
export TMPDIR="$VSC_SCRATCH/pip_tmp"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
```

---

## 5. Stage 0: convert the manually annotated source to XML

### Script

```text
1A_convert2xml.py
```

### When it is needed

Run this only when starting from the manually annotated semicolon-separated `Pride and Prejudice` file. If a clean XML source already exists, this stage can be skipped.

### What it does

The converter:

1. reads `input/pride_and_prejudice_JK.csv`;
2. identifies narrative text and turn boundaries represented by `$` and `$$` markers;
3. standardizes missing speaker and addressee values as `Unknown`;
4. converts bystander information to `yes` or `no`;
5. adds `indirectspeech="yes"` where the source annotation marks indirect speech;
6. reconstructs direct speech and reporting clauses;
7. writes `output/pride_and_prejudice_clean.xml`.

### Run

```bash
python 1A_convert2xml.py
```

### Validate

```bash
xmllint --noout output/pride_and_prejudice_clean.xml
```

This conversion embeds fixed input and output paths and a fixed book title. Change those constants before reusing it for another source.

---

## 6. Stage 1: create training and validation examples

### Script

```text
1_prepare_data.py
```

### Input

```text
output/pride_and_prejudice_clean.xml
```

### Outputs

```text
intermediate_output/train.jsonl
intermediate_output/val.jsonl
```

### Design

Each XML paragraph becomes one chat training example with three messages:

1. a fixed system prompt describing the annotation task;
2. a user message containing up to three preceding raw paragraphs and the raw target paragraph;
3. the assistant target containing the same target paragraph with `<turn>` and `<interruption>` markup.

The user message has the following stable structure:

```text
Context history:
<up to three previous paragraphs>

Target paragraph to parse:
<current raw paragraph>
```

At the beginning of the book, the context is:

```text
No preceding context (Start of book).
```

Turn number attributes are removed before training. Numbering is added only during full-book inference. The script validates that no turn-number attributes remain in either split.

### Run

```bash
python 1_prepare_data.py
```

### Important methodological limitation

The current script randomly splits paragraph examples after context has already been attached. Consequently, adjacent paragraphs can end up in different splits while their text appears in each other's context. This is reproducible because `random_state=42` is fixed, but it is not a strict document- or chapter-independent evaluation. For a future study, split by book or chapter before constructing context windows.

---

## 7. Stage 2: validate preprocessing before allocating GPUs

The maintained 70B trainer can run all data, tokenization and collator checks without loading the 70B model.

```bash
python 2_train_llama70B.py --preprocess-only
```

This checks that:

- the train and validation files exist;
- every example ends with an assistant target;
- examples fit within the 2,048-token limit;
- long or invalid examples are audited and excluded;
- the Llama assistant header is found without hard-coded token IDs;
- loss labels are active only for the assistant answer;
- system and user prompt tokens are masked with `-100`;
- dynamic padding also masks padded labels with `-100`;
- every retained example has at least one target token.

The exclusion report is written to:

```text
intermediate_output/long_examples_audit.json
```

Run this stage after any change to the data preparation, tokenizer, chat template or maximum sequence length.

---

## 8. Stage 3: GPU smoke test

### SLURM script

```text
janeausten_70B_smoketest_job.sh
```

### Purpose

The smoke test verifies distributed model loading, LoRA injection, DeepSpeed integration, BF16 parameter dtypes and two optimizer steps on small train and validation subsets. It intentionally does not save a final adapter.

### Resources

The supplied job requests:

- one node;
- four H100 GPUs;
- 16 CPU cores;
- 180 GB system memory;
- 45 minutes.

### Submit

```bash
sbatch janeausten_70B_smoketest_job.sh
```

### Monitor

```bash
squeue -u "$USER"
tail -f logs_austen_70B.<job-id>.out
```

A successful run ends with a message stating that the GPU smoke test completed and final model saving was skipped.

---

## 9. Stage 4: train the maintained 70B LoRA adapter

### Scripts and configuration

```text
2_train_llama70B.py
deepspeed_config.json
janeausten_70B_job.sh
```

### Model and method

The trainer uses:

- `meta-llama/Llama-3.3-70B-Instruct`;
- BF16;
- Flash Attention 2;
- LoRA on `q_proj`, `k_proj`, `v_proj` and `o_proj`;
- LoRA rank 16 and alpha 32;
- gradient checkpointing;
- assistant-only loss masking;
- dynamic batch padding;
- DeepSpeed ZeRO stage 3;
- CPU optimizer offload;
- four distributed GPU processes launched with Accelerate.

The production configuration uses a micro-batch size of one per GPU, gradient accumulation of 16 and three epochs. The effective global batch size is printed at runtime.

### Submit

```bash
sbatch janeausten_70B_job.sh
```

### Output

The trained adapter and tokenizer are written to:

```text
$VSC_SCRATCH/JaneAusten/fine_tuned_austen_70b_tokenfix_v5
```

Intermediate checkpoints are written under:

```text
$VSC_SCRATCH/JaneAusten/results_austen_70b_tokenfix_v5
```

Checkpoint resumption is disabled in the current Python code through `ALLOW_CHECKPOINT_RESUME = False`. Do not assume that rerunning the production job will resume automatically unless this setting is deliberately changed and tested.

### DeepSpeed configuration

`deepspeed_config.json` enables BF16 and ZeRO stage 3, offloads optimizer state to pinned CPU memory, and lets the Trainer determine accumulation and batch geometry automatically.

---

## 10. Stage 5: evaluate the adapter

### Scripts

```text
3_evaluate.py
janeausten_eval_70B_job.sh
```

### Purpose

The evaluator loads the base model and LoRA adapter across two H100 GPUs, generates predictions for the validation examples and compares the ordered turn metadata against the reference targets.

It reports:

- exact paragraph metadata matches;
- exact turn-count matches;
- total true and predicted turns;
- missing turns;
- extra turns;
- classification reports for speaker, addressee and bystanders.

Detailed per-paragraph output is written to:

```text
$VSC_SCRATCH/JaneAusten/evaluation_val_70b.jsonl
```

### Submit

```bash
sbatch janeausten_eval_70B_job.sh
```

### Interpretation caveat

Turns are aligned by order rather than by optimal span matching. One missed or extra turn can shift subsequent comparisons inside the same paragraph. Entity aliases are also scored literally, so `Jane` and `Miss Bennet` count as different labels unless normalized separately.

---

## 11. Stage 6: full-book inference

### Script

```text
4_inference.py
```

### Architecture

The inference program:

1. loads the tokenizer from the adapter directory;
2. distributes the 70B base model statically over the visible GPUs;
3. attaches the LoRA adapter;
4. reads each plain-text book as blank-line-separated paragraphs;
5. preserves Gutenberg front matter and back matter but excludes them from model inference;
6. starts literary inference at the first chapter, volume or book heading where available;
7. skips structural headings;
8. uses the exact system prompt and user-message structure used during fine-tuning;
9. provides the preceding three paragraphs as context by default;
10. generates deterministically with `do_sample=False`;
11. parses turns, attribute aliases for indirect speech, and interruptions;
12. computes minimum generated-token probabilities for the turn and attributes;
13. maps generated turn spans back to the original source paragraph;
14. writes only original source text into the final XML;
15. records failed mappings rather than forcing altered model text into the source;
16. writes an audit JSONL entry for every paragraph;
17. saves atomic XML checkpoints and can resume from `processed_paragraphs`.

### Why source mapping matters

The model may modify punctuation, wording, emphasis or reporting clauses. The pipeline therefore treats model output as a proposed annotation structure, not as replacement text. A turn is inserted only when its de-tagged content maps back to the original paragraph. Otherwise, the source paragraph remains unchanged and receives `annotation_status="mapping_failed"` or `partially_mapped`.

### Confidence fields

The output stores:

- `confidence`;
- `turn_detection_confidence`;
- `S_confidence`;
- `A_confidence`;
- `bystanders_confidence`;
- `indirectspeech_confidence`;
- `indirectspeech_confidence_basis`.

These values are relative model scores, not calibrated probabilities. They should be used to prioritize manual review, not interpreted as an empirical probability that an annotation is correct.

### Generic smoke test

The generic job `janeausten_infer_70B_job.sh` processes the first 120 paragraphs of `sense_and_sensibility.txt` with `--overwrite`. It is a test job, not a production full-book job.

```bash
sbatch janeausten_infer_70B_job.sh
```

### Full-book jobs

The supplied jobs request two H100 GPUs per book:

```bash
sbatch janeausten_infer_70B_lady_susan_job.sh
sbatch janeausten_infer_70B_northanger_abbey_job.sh
sbatch janeausten_infer_70B_persuasion_job.sh
sbatch janeausten_infer_70B_sense_and_sensibility_job.sh
sbatch janeausten_infer_70B_emma_job.sh
sbatch janeausten_infer_70B_mansfield_park_job.sh
```

Each job differs mainly in `--file_pattern`, walltime, job name and log name.

### Outputs per book

```text
output/<book>_annotated_llama70B.xml
output/<book>_annotated_llama70B_audit.jsonl
```

### Resuming an interrupted run

Rerun the same inference command without `--overwrite`. The program reads `processed_paragraphs` from the existing XML and continues with the next paragraph. The audit JSONL is appended.

Do not use `--overwrite` for resumption. It deletes the existing audit and starts the XML again from paragraph one.

### Validate a finished book

```bash
xmllint --noout output/emma_annotated_llama70B.xml

grep -n 'annotation_status' output/emma_annotated_llama70B.xml | head

grep -n 'review_recommended="true"' output/emma_annotated_llama70B.xml | head
```

### Genre limitation

`Lady Susan` is epistolary. A paragraph-level dialogue model does not consistently propagate the letter writer and recipient through all paragraphs of a letter. Treat this result as preliminary unless a separate letter-structure parser is added.

---

## 12. Stage 7: normalize character labels

### Files

```text
5_normalize_jane_austen_xml.py
jane_austen_character_normalization_v2.json
```

### Purpose

The LLM may refer to the same character using a first name, title, married name or surname. Exact alias normalization maps safe variants to one canonical label while preserving the raw prediction.

Example:

```xml
<turn S="George Knightley" S_raw="Mr. Knightley"
      A="Emma Woodhouse" A_raw="Emma">
```

### Run

```bash
python 5_normalize_jane_austen_xml.py \
  output/emma_annotated_llama70B.xml \
  --dictionary jane_austen_character_normalization_v2.json \
  --output output/normalised/emma_annotated_normalized.xml
```

### Outputs

```text
output/normalised/emma_annotated_normalized.xml
output/normalised/emma_annotated_normalized.normalization_audit.jsonl
output/normalised/emma_annotated_normalized.normalization_summary.json
```

The postprocessor:

- never overwrites the input file;
- normalizes only exact aliases;
- stores changed original labels as `S_raw` and `A_raw`;
- preserves text and interruptions;
- lists unresolved labels by frequency.

Do not add context-dependent labels such as `he`, `she`, `aunt`, `cousin`, `Unknown`, `Mr. Bertram`, `Tilney` or `Mr. Ferrars` to a flat global mapping unless their identity is unambiguous in that book.

---

## 13. Recommended execution sequence

### From a new manually annotated source

```bash
python 1A_convert2xml.py
xmllint --noout output/pride_and_prejudice_clean.xml
python 1_prepare_data.py
python 2_train_llama70B.py --preprocess-only
sbatch janeausten_70B_smoketest_job.sh
sbatch janeausten_70B_job.sh
sbatch janeausten_eval_70B_job.sh
```

After training and evaluation, submit the desired full-book inference jobs, validate their XML, and run normalization.

### When the trained adapter already exists

Skip conversion, preparation and training:

```bash
sbatch janeausten_eval_70B_job.sh
sbatch janeausten_infer_70B_emma_job.sh
```

Then validate and normalize the resulting XML.

### When only the normalization dictionary changes

Do not retrain and do not repeat inference. Rerun only stage 7 against the original annotated XML.

---

## 14. SLURM operations

Submit:

```bash
sbatch job_script.sh
```

List jobs:

```bash
squeue -u "$USER"
```

Inspect accounting after completion:

```bash
sacct -j <job-id> --format=JobID,JobName,State,Elapsed,Timelimit,AllocTRES,MaxRSS,ExitCode
```

Follow a log:

```bash
tail -f logs_austen_70B_infer_emma_<job-id>.out
```

Cancel:

```bash
scancel <job-id>
```

Before submitting a job, validate the Python syntax and shell syntax:

```bash
python -m py_compile 2_train_llama70B.py 3_evaluate.py 4_inference.py 5_normalize_jane_austen_xml.py
bash -n janeausten_70B_job.sh
bash -n janeausten_infer_70B_emma_job.sh
```

---

## 15. Known limitations and future improvements

1. Training currently comes from one complete novel. A future training set should sample chapters and difficult cases across multiple books and genres.
2. The fixed three-paragraph context is unstable because paragraph length varies. A future version should use a minimum word or token budget with an upper limit, identically during training and inference.
3. The current random paragraph split can leak neighboring text through contextual windows. Split by chapter or book first.
4. Indirect speech is underrepresented and encoded asymmetrically, because positive examples contain `indirectspeech="yes"` while negative examples usually omit the attribute.
5. Free indirect speech does not fit cleanly into the current binary schema.
6. Interruption detection sometimes marks emphasis or spoken clauses rather than reporting narration.
7. Exact alias normalization improves consistency but cannot correct an incorrect speaker assignment.
8. Contextual entity resolution is still needed for `Unknown`, pronouns, relation terms and book-external hallucinated names.
9. The supplied job files install packages during jobs and contain authentication material. For reproducible use, move dependency installation to environment setup, pin versions, and keep credentials outside version control.

---

## 16. Reproducibility checklist

Record the following for every experiment:

- Git commit hash;
- base model identifier and revision;
- adapter output path;
- Hugging Face Transformers, PEFT, Accelerate, Datasets, DeepSpeed and PyTorch versions;
- CUDA and driver versions;
- SLURM job ID and resource allocation;
- training and validation file hashes;
- random seed and split procedure;
- `deepspeed_config.json`;
- generated long-example audit;
- evaluation JSONL and console metrics;
- inference XML and audit JSONL;
- normalization dictionary version and normalization audit.

This separates the immutable source text, model proposals, accepted source mappings and later character normalization into traceable layers.

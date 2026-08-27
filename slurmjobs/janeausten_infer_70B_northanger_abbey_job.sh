#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pieter.bolle@student.kuleuven.be
#SBATCH --job-name=austen_nlp_70B_infer_northanger_abbey
#SBATCH --cluster=wice           # Target the WICE cluster
#SBATCH --partition=gpu_h100          # Request the GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8       # Meer CPU cores voor data-sharding
#SBATCH --mem=180G            # Meer systeemgeheugen
#SBATCH --time=04:00:00
#SBATCH --output=logs_austen_70B_infer_northanger_abbey_%j.out
#SBATCH --account=lp_edu_maibi_anndl

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DISABLE_P2P=1

# --- VERWIJS HUGGINGFACE NAAR DE SCRATCH PARTITIE ---
export HF_HOME=$VSC_SCRATCH/huggingface_cache
# Voor oudere versies van transformers (optioneel, voor de zekerheid):
export TRANSFORMERS_CACHE=$VSC_SCRATCH/huggingface_cache
export HF_TOKEN="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"

export TMPDIR=$VSC_SCRATCH/pip_tmp
mkdir -p $TMPDIR
export TRITON_CACHE_DIR=$VSC_SCRATCH/triton_cache

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_TIMEOUT=7200

# Modules laden (afhankelijk van specifieke VSC-architectuur)
module purge                            # Schoon eventuele conflicterende modules op
module load cluster/wice/gpu
module load foss/2022b                  # 1. Laad de hoofdtoolchain (bevat GCCcore-12.2.0)
module load Python/3.10.8-GCCcore-12.2.0 # 2. Laad Python 3.10
module load CUDA/12.1.1                 # 3. Laad CUDA voor de H100 GPU

# Virtuele omgeving activeren
cd /vsc-hard-mounts/leuven-data/381/vsc38154/JaneAusten/
source venv/bin/activate
pip install --upgrade pip
pip install wheel setuptools ninja
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flash_attn --no-build-isolation --no-cache-dir

rm -rf $TMPDIR

# Stap 2: Model evalueren
python 4_inference.py --base_model_id meta-llama/Llama-3.3-70B-Instruct --adapter_dir "$VSC_SCRATCH/JaneAusten/fine_tuned_austen_70b_tokenfix_v5" --input_dir input --output_dir output --file_pattern "northanger_abbey.txt" --max_new_tokens 1024 --checkpoint_every 5 --verbose
